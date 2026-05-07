"""
LOCAL WINDOWS GPU VERSION - WITH VERTEX AI GEMINI
==================================================
All paths point to C:\\Opeyemi\\PROMPTS\\
Gemini accessed via Vertex AI service account (not bare API key).

Run end-to-end. No Colab/Drive dependencies.

FIXES APPLIED (2026-05-04):
  - Removed `temperature=0.0` from call_claude_judge() because
    claude-opus-4-7 deprecated the temperature parameter and was
    returning 400 errors on every call. Pattern matches the working
    CLAUDE_ReACT_Final_PARALLEL.ipynb generation code.
  - Added GenerateContentConfig + 60s rate-limit sleep to call_gemini_judge()
    matching the working Gemini_LEAST-MOST_Final_PARALLEL.ipynb pattern.
  - Added automatic cleanup of bad checkpoint entries on script start.
"""

import os
import json
import re
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm.auto import tqdm    # auto-detects Jupyter vs terminal; renders bar correctly in notebooks

# Force any buffered output to flush so progress bars and prints appear immediately
sys.stdout.flush()


# =====================================================================
# LOCAL WINDOWS PATHS
# =====================================================================
RESULTS_BASE = r'C:\Opeyemi\PROMPTS\RESULTS'
GT_PATH      = r'C:\Opeyemi\PROMPTS\ANNOTATION'
OUTPUT_DIR   = r'C:\Opeyemi\PROMPTS\EVALUATION'
API_KEYS_DIR = r'C:\Opeyemi\PROMPTS\API-KEYS'

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_PATHS = {
    'Claude': os.path.join(RESULTS_BASE, 'CLAUDE'),
    'GPT':    os.path.join(RESULTS_BASE, 'GPT'),
    'Gemini': os.path.join(RESULTS_BASE, 'GEMINI'),
}

# Cache and output file paths
GT_CACHE_PATH        = os.path.join(OUTPUT_DIR, 'gt_data_cache.json')
TRIPLETS_CACHE_PATH  = os.path.join(OUTPUT_DIR, 'triplets_stratified.csv')
PANEL_CHECKPOINT     = os.path.join(OUTPUT_DIR, 'panel_checkpoint_subset.json')
PANEL_RAW_PATH       = os.path.join(OUTPUT_DIR, 'panel_raw_judge_labels_subset.csv')
JUDGE_LABELS_PATH    = os.path.join(OUTPUT_DIR, 'llm_judge_labels_subset.csv')
JUDGE_AGREEMENT_PATH = os.path.join(OUTPUT_DIR, 'inter_judge_agreement_subset.json')
FULL_LABELED_PATH    = os.path.join(OUTPUT_DIR, 'full_labeled_dataset_subset.csv')
CLASSIFIER_RESULTS   = os.path.join(OUTPUT_DIR, 'retrained_classifier_results.json')
HALLUCINATION_MATRIX = os.path.join(OUTPUT_DIR, 'hallucination_matrix.json')
SIGNATURES_PATH      = os.path.join(OUTPUT_DIR, 'hallucination_signatures.json')


# =====================================================================
# API CREDENTIALS
# =====================================================================
def load_key(filename):
    with open(os.path.join(API_KEYS_DIR, filename), 'r') as f:
        return f.read().strip()

ANTHROPIC_API_KEY = load_key('claude.txt')
OPENAI_API_KEY    = load_key('chatgpt.txt')

# Gemini via Vertex AI (service account, not bare API key)
# IMPORTANT: location must be 'global' for gemini-3.1-pro-preview access
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(
    API_KEYS_DIR, 'multi-object-tracking-491921-946a152255ed.json'
)
os.environ['GOOGLE_CLOUD_PROJECT']      = 'multi-object-tracking-491921'
os.environ['GOOGLE_CLOUD_LOCATION']     = 'global'
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'


# =====================================================================
# RUN SETTINGS
# =====================================================================
TECHNIQUES = {
    'ZERO':          'Zero-Shot',
    'SEQUENTIAL':    'Sequential',
    'LEAST-TO-MOST': 'Least-to-Most',
    'REACT':         'ReAct',
}

ALL_JUDGES = ['Claude', 'GPT', 'Gemini']

# ----------------------------------------------------------------------
# JUDGE_FILTER: control which judges run in this pass
#   None              = all enabled judges run (cross-judging: 2 per row)
#   ['GPT']           = ONLY GPT judges (fastest path to test)
#   ['Claude']        = ONLY Claude judges
#   ['Gemini']        = ONLY Gemini judges
#   ['GPT', 'Claude'] = both, skip Gemini
# ----------------------------------------------------------------------
JUDGE_FILTER = None     # use all 3 judges (with no-self-evaluation)

def judges_for(model_name):
    """Return cross-judges for a given generating model, intersected with JUDGE_FILTER."""
    cross = [j for j in ALL_JUDGES if j != model_name]
    if JUDGE_FILTER is not None:
        cross = [j for j in cross if j in JUDGE_FILTER]
    return cross

HTYPES = ['SCENE_FABRICATION', 'CRIME_MISCLASSIFICATION', 'CRIME_MISSED',
          'SEVERITY_MINIMIZATION', 'ENTITY_FABRICATION', 'PHANTOM_ACTORS']
HTYPE_MAP = {
    'SCENE_FABRICATION':      'H1',
    'CRIME_MISCLASSIFICATION':'H2',
    'CRIME_MISSED':           'H3',
    'SEVERITY_MINIMIZATION':  'H4',
    'ENTITY_FABRICATION':     'H5',
    'PHANTOM_ACTORS':         'H6',
}


# =====================================================================
# SANITY CHECKS
# =====================================================================
assert os.path.isdir(RESULTS_BASE), f'Results base missing: {RESULTS_BASE}'
assert os.path.isdir(GT_PATH),      f'Annotation folder missing: {GT_PATH}'
assert os.path.isdir(API_KEYS_DIR), f'API keys folder missing: {API_KEYS_DIR}'
assert os.path.isfile(os.environ['GOOGLE_APPLICATION_CREDENTIALS']), \
    f"Vertex credentials missing: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}"

print('=' * 70)
print('Configuration')
print('=' * 70)
print(f'Results base:     {RESULTS_BASE}')
print(f'Ground truth:     {GT_PATH}')
print(f'Output dir:       {OUTPUT_DIR}')
print(f'API keys dir:     {API_KEYS_DIR}')
print(f'Anthropic key:    {"loaded" if ANTHROPIC_API_KEY else "MISSING"}')
print(f'OpenAI key:       {"loaded" if OPENAI_API_KEY else "MISSING"}')
print(f'Vertex AI:        project={os.environ["GOOGLE_CLOUD_PROJECT"]}')
print(f'                  location={os.environ["GOOGLE_CLOUD_LOCATION"]}')
print(f'Techniques:       {list(TECHNIQUES.values())}')
print(f'Panel:            {ALL_JUDGES}')
print(f'Active judges:    {JUDGE_FILTER if JUDGE_FILTER else "ALL"}')
print()


# =====================================================================
# AUTO-CLEAN BAD CHECKPOINT ENTRIES (Claude 400 errors + any -1 errors)
# =====================================================================
# Older runs may have saved bad entries. Clean them so they get retried.
print('=' * 70)
print('Auto-cleaning bad checkpoint entries (if any)')
print('=' * 70)

if os.path.exists(PANEL_CHECKPOINT):
    with open(PANEL_CHECKPOINT) as f:
        _ckpt = json.load(f)

    _initial = len(_ckpt)

    # Detect Claude entries with the deprecated-temperature 400 error,
    # and any entry that has -1 labels (meaning it errored).
    _bad_keys = []
    for k, v in _ckpt.items():
        reasoning = v.get('reasoning', '') if isinstance(v, dict) else ''
        has_neg1  = any(v.get(h, 0) == -1 for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']) if isinstance(v, dict) else False
        is_claude_temp_err = (
            k.endswith('_Claude')
            and 'temperature' in reasoning.lower()
            and 'deprecated' in reasoning.lower()
        )
        if is_claude_temp_err or has_neg1:
            _bad_keys.append(k)

    for k in _bad_keys:
        del _ckpt[k]

    if _bad_keys:
        with open(PANEL_CHECKPOINT, 'w') as f:
            json.dump(_ckpt, f)
        print(f'  Removed {len(_bad_keys)} bad entries')
        print(f'  Checkpoint: {_initial} -> {len(_ckpt)} entries')

        # Sample the kinds of errors removed
        from collections import Counter as _Counter
        _err_kinds = _Counter()
        # Re-scan original (we already deleted from _ckpt, so just summarize)
        _err_kinds['removed'] = len(_bad_keys)
        print(f'  Will be retried on this run.')
    else:
        print(f'  No bad entries found ({_initial} entries clean).')
else:
    print('  No existing checkpoint — starting fresh.')
print()


# =====================================================================
# GROUND TRUTH LOADER
# =====================================================================
GT_FILES = ['UCFCrime_Train.json', 'UCFCrime_Val.json', 'UCFCrime_Test.json']


def derive_crime_type(video_id):
    name = video_id.replace('_x264', '')
    m = re.match(r'^([A-Za-z]+?)\d', name)
    return m.group(1) if m else 'Unknown'


def is_anomalous(video_id):
    return not video_id.startswith('Normal_')


def parse_ground_truth(gt_base_path):
    gt_data = {}
    n_dropped = 0
    for fname in GT_FILES:
        fpath = os.path.join(gt_base_path, fname)
        if not os.path.exists(fpath):
            print(f'  WARN: missing {fname}')
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        kept = 0
        for video_id, info in data.items():
            if not is_anomalous(video_id):
                n_dropped += 1
                continue
            gt_data[video_id] = {
                'crime_type': derive_crime_type(video_id),
                'sentences':  info.get('sentences', []),
                'timestamps': info.get('timestamps', []),
                'duration':   info.get('duration'),
            }
            kept += 1
        print(f'  {fname}: kept {kept:,} anomalous')
    print(f'Excluded {n_dropped:,} Normal videos')
    return gt_data


print('=' * 70)
print('Loading ground truth')
print('=' * 70)
if os.path.exists(GT_CACHE_PATH):
    with open(GT_CACHE_PATH) as f:
        gt_data = json.load(f)
    print(f'Loaded GT from cache: {len(gt_data):,} videos')
else:
    gt_data = parse_ground_truth(GT_PATH)
    with open(GT_CACHE_PATH, 'w') as f:
        json.dump(gt_data, f)
    print(f'Cached {len(gt_data):,} anomalous-only annotations')

print(f'\nVideos by crime type:')
for crime, n in sorted(Counter(v['crime_type'] for v in gt_data.values()).items()):
    print(f'  {crime:<20} {n:>5}')
print()


# =====================================================================
# VIDEO ID EXTRACTION FROM FILENAME
# =====================================================================
GEMINI_VID_RE = re.compile(r'^([A-Za-z]+_[A-Za-z]+\d+(?:_x264)?)')

def video_id_from_filename(filename):
    m = GEMINI_VID_RE.match(filename)
    return m.group(1) if m else None


def resolve_gt_key(video_id, gt_data):
    candidates = [
        video_id,
        video_id.replace('_x264', ''),
    ]
    parts = video_id.split('_')
    if len(parts) >= 2:
        without_prefix = '_'.join(parts[1:])
        candidates.append(without_prefix)
        candidates.append(without_prefix.replace('_x264', ''))
    for c in candidates:
        if c in gt_data:
            return c
    return None


# =====================================================================
# EXTRACTION HELPERS
# =====================================================================

def extract_claude_zero(entry):       return entry.get('final_analysis', '')
def extract_claude_sequential(entry): return entry.get('stages', {}).get('stage4_final_summary', '')
def extract_claude_ltm(entry):        return entry.get('final_analysis', '')
def extract_claude_react(entry):      return entry.get('final_analysis', '')
def extract_gpt_zero(entry):          return entry.get('final_analysis', '')


def extract_gpt_sequential(entry):
    stages = entry.get('stages', {})
    val = stages.get('stage4_final_summary', '')
    if isinstance(val, str) and val.strip():
        return val.strip()
    val = stages.get('Final Synthesis', '')
    if isinstance(val, dict):   return val.get('response', '')
    if isinstance(val, str):    return val.strip()
    if stages:
        last = list(stages.values())[-1]
        if isinstance(last, str):  return last
        if isinstance(last, dict): return last.get('response', '')
    return ''


def extract_gpt_ltm(entry):
    stages = entry.get('stages', {})
    if not stages:
        return ''
    level_keys = sorted(
        [k for k in stages if k.lower().startswith('level')],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
        reverse=True
    )
    if not level_keys:
        level_keys = sorted(stages.keys(), reverse=True)
    val = stages[level_keys[0]]
    if isinstance(val, str):   return val
    if isinstance(val, list):  return '\n'.join(str(v) for v in val)
    if isinstance(val, dict):  return val.get('response', val.get('final_report', str(val)))
    return ''


def extract_gpt_react(entry):
    react_log = entry.get('react_log', {})
    if isinstance(react_log, str):   return react_log
    if isinstance(react_log, dict):  return '\n\n'.join(str(v) for v in react_log.values())
    if isinstance(react_log, list):  return '\n\n'.join(str(v) for v in react_log)
    return entry.get('final_answer', '')


def extract_gemini_sequential_obj(obj):
    results = obj.get('sequential_results', obj)
    if not isinstance(results, dict): return ''
    final = results.get('Final Synthesis', {})
    if isinstance(final, dict):
        text = final.get('response', '')
        if text and text.strip(): return text.strip()
    if isinstance(final, str) and final.strip(): return final.strip()
    step_keys = sorted(
        [k for k in results if k.startswith('Step')],
        key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0,
        reverse=True
    )
    if step_keys:
        val = results[step_keys[0]]
        if isinstance(val, dict): return val.get('response', '')
        return str(val)
    return ''


def extract_gemini_ltm_obj(obj):
    results = obj.get('least_to_most_results', obj)
    if not isinstance(results, dict): return ''
    step_keys = sorted(
        [k for k in results if k.startswith('Step')],
        key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0,
        reverse=True
    )
    if not step_keys: return ''
    last = results[step_keys[0]]
    if isinstance(last, dict): return last.get('response', '')
    return last if isinstance(last, str) else ''


def extract_gemini_react_chunk_obj(obj):
    for key, val in obj.items():
        if isinstance(val, dict):
            text = val.get('react_analysis', '')
            if text and text.strip(): return text.strip()
        if key == 'react_analysis' and isinstance(val, str):
            return val.strip()
    return ''


# =====================================================================
# LOADERS
# =====================================================================

def load_summary_outputs(model, tech_folder):
    tech_path = os.path.join(BASE_PATHS[model], tech_folder)
    if not os.path.isdir(tech_path):
        return {}
    all_json = sorted([f for f in os.listdir(tech_path)
                       if f.endswith('.json') and 'checkpoint' not in f.lower()])
    summary_files = [f for f in all_json if 'summary' in f.lower()]
    if not summary_files:
        summary_files = all_json
    if not summary_files:
        return {}
    sf = os.path.join(tech_path, summary_files[-1])
    try:
        with open(sf, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f'  WARN: cannot read {sf}: {e}')
        return {}
    extractor_map = {
        ('Claude', 'ZERO'):          extract_claude_zero,
        ('Claude', 'SEQUENTIAL'):    extract_claude_sequential,
        ('Claude', 'LEAST-TO-MOST'): extract_claude_ltm,
        ('Claude', 'REACT'):         extract_claude_react,
        ('GPT',    'ZERO'):          extract_gpt_zero,
        ('GPT',    'SEQUENTIAL'):    extract_gpt_sequential,
        ('GPT',    'LEAST-TO-MOST'): extract_gpt_ltm,
        ('GPT',    'REACT'):         extract_gpt_react,
    }
    extractor = extractor_map.get((model, tech_folder))
    if extractor is None:
        print(f'  WARN: no extractor for ({model}, {tech_folder})')
        return {}
    outputs = {}
    for video_id, entry in data.items():
        if not isinstance(entry, dict):
            continue
        text = extractor(entry)
        if text and len(text.strip()) >= 50:
            outputs[video_id] = text.strip()
    return outputs


def load_gemini_outputs(tech_folder):
    tech_path = os.path.join(BASE_PATHS['Gemini'], tech_folder)
    if not os.path.isdir(tech_path):
        return {}
    all_files = sorted([f for f in os.listdir(tech_path)
                        if f.endswith('.json') and '_checkpoints' not in f])
    per_video = defaultdict(list)

    if tech_folder == 'ZERO':
        for fname in all_files:
            try:
                with open(os.path.join(tech_path, fname), encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception:
                continue
            analysis = obj.get('Analysis', obj)
            text = analysis.get('answer', '') if isinstance(analysis, dict) else ''
            if not text: text = obj.get('answer', '')
            vid = video_id_from_filename(fname)
            if vid and text and text.strip():
                per_video[vid].append(text.strip())
        return {vid: '\n\n=== NEXT CHUNK ===\n\n'.join(t) for vid, t in per_video.items()}

    elif tech_folder == 'SEQUENTIAL':
        for fname in [f for f in all_files if '_complete_' in f]:
            try:
                with open(os.path.join(tech_path, fname), encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception:
                continue
            text = extract_gemini_sequential_obj(obj)
            vid  = video_id_from_filename(fname)
            if vid and text and text.strip():
                per_video[vid].append(text.strip())
        return {vid: texts[-1] for vid, texts in per_video.items()}

    elif tech_folder == 'LEAST-TO-MOST':
        for fname in [f for f in all_files if '_complete_' in f]:
            try:
                with open(os.path.join(tech_path, fname), encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception:
                continue
            text = extract_gemini_ltm_obj(obj)
            vid  = video_id_from_filename(fname)
            if vid and text and text.strip():
                per_video[vid].append(text.strip())
        return {vid: texts[-1] for vid, texts in per_video.items()}

    elif tech_folder == 'REACT':
        for fname in all_files:
            try:
                with open(os.path.join(tech_path, fname), encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception:
                continue
            text = extract_gemini_react_chunk_obj(obj)
            vid  = video_id_from_filename(fname)
            if vid and text and text.strip():
                per_video[vid].append(text.strip())
        return {vid: '\n\n=== NEXT CHUNK ===\n\n'.join(t) for vid, t in per_video.items()}

    return {}


# =====================================================================
# TRIPLET BUILDER
# =====================================================================

def build_triplets(gt_data):
    rows = []
    print('Building triplets...')
    for model in ['Claude', 'GPT', 'Gemini']:
        for tech_folder, tech_label in TECHNIQUES.items():
            print(f'  [{model}/{tech_label}]', end=' ', flush=True)
            outputs = (load_gemini_outputs(tech_folder) if model == 'Gemini'
                       else load_summary_outputs(model, tech_folder))
            n = 0
            for video_id, output_text in outputs.items():
                gt_key = resolve_gt_key(video_id, gt_data)
                if gt_key is None:
                    continue
                gt_info = gt_data[gt_key]
                rows.append({
                    'model':                 model,
                    'technique':             tech_label,
                    'video':                 gt_key,
                    'crime_type':            gt_info['crime_type'],
                    'ground_truth':          ' '.join(gt_info['sentences']),
                    'model_output':          output_text,
                    'model_output_full_len': len(output_text),
                })
                n += 1
            print(f'{n} triplets')
    return pd.DataFrame(rows)


def load_triplets_cache(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        required = {'model', 'technique', 'video', 'crime_type', 'ground_truth', 'model_output'}
        if df.empty or not required.issubset(df.columns):
            print('Cache invalid — rebuilding.')
            os.remove(path)
            return None
        return df
    except Exception as e:
        print(f'Cache unreadable ({e}) — rebuilding.')
        os.remove(path)
        return None


print('=' * 70)
print('Building triplets')
print('=' * 70)

df_triplets = load_triplets_cache(TRIPLETS_CACHE_PATH)
if df_triplets is None:
    df_triplets = build_triplets(gt_data)
    if df_triplets.empty:
        raise RuntimeError('build_triplets() returned 0 rows.')
    df_triplets.to_csv(TRIPLETS_CACHE_PATH, index=False)
    print(f'\nCached {len(df_triplets):,} triplets')
else:
    print(f'Loaded from cache: {len(df_triplets):,} rows')

print(f'\nTotal triplets: {len(df_triplets):,}')
print(f'\nPer (model, technique):')
print(df_triplets.groupby(['model', 'technique']).size().unstack(fill_value=0))
print(f'\nPer model:')
print(df_triplets.groupby('model').size())
print(f'\nMean output length (chars):')
print(df_triplets.groupby(['model', 'technique'])['model_output_full_len']
      .mean().unstack(fill_value=0).round(0).astype(int))


# =====================================================================
# LLM JUDGE PANEL
# =====================================================================

import anthropic
import openai
from google import genai as google_genai
from google.genai import types as google_genai_types

# Initialise API clients
_anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
_openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
_genai_client     = google_genai.Client(            # explicit Vertex AI init
    vertexai=True,
    project=os.environ['GOOGLE_CLOUD_PROJECT'],
    location=os.environ['GOOGLE_CLOUD_LOCATION'],   # 'global' for gemini-3.1-pro-preview
)


# ------------------------------------------------------------------
# Judge prompt
# ------------------------------------------------------------------
JUDGE_SYSTEM = """You are an expert forensic video analysis evaluator.
Your task is to detect hallucinations in an AI model's analysis of a crime video.
You will be given:
  1. GROUND TRUTH: the human-annotated description of what actually happened.
  2. MODEL OUTPUT: the AI system's analysis of the same video.

Evaluate whether the MODEL OUTPUT contains any of the following hallucination types.
For each type respond ONLY with 0 (not present) or 1 (present).

Hallucination types:
  H1 SCENE_FABRICATION    - invents setting details not in ground truth (wrong location, objects, environment)
  H2 CRIME_MISCLASSIFICATION - identifies the wrong crime type (e.g. says Robbery when it is Assault)
  H3 CRIME_MISSED         - fails to mention the primary crime that is clearly described in ground truth
  H4 SEVERITY_MINIMIZATION - describes crime as less serious than ground truth indicates
  H5 ENTITY_FABRICATION   - invents people, vehicles, or objects not present in ground truth
  H6 PHANTOM_ACTORS       - adds perpetrators or victims not mentioned in ground truth

Respond ONLY with a JSON object in this exact format, no extra text:
{
  "H1": 0,
  "H2": 0,
  "H3": 0,
  "H4": 0,
  "H5": 0,
  "H6": 0,
  "reasoning": "brief one-sentence explanation"
}"""


def make_judge_prompt(ground_truth: str, model_output: str) -> str:
    # Truncate very long outputs to stay within context limits
    gt_trunc  = ground_truth[:1500]
    out_trunc = model_output[:3000]
    return (
        f"GROUND TRUTH:\n{gt_trunc}\n\n"
        f"MODEL OUTPUT:\n{out_trunc}\n\n"
        "Now evaluate for hallucinations and return the JSON."
    )


# ------------------------------------------------------------------
# Per-judge call functions with exponential back-off
# ------------------------------------------------------------------

def _retry_call(fn, max_retries=7, base_delay=4.0, max_delay=300.0):
    """
    Wrap any API call with exponential back-off on rate-limit / server errors.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            err_str  = str(e)
            err_low  = err_str.lower()
            is_rate   = any(x in err_low for x in ['429', 'rate limit', 'quota',
                                                    'resource exhausted', 'resource_exhausted'])
            is_server = any(x in err_low for x in ['500', '502', '503', '504',
                                                    'server error', 'unavailable', 'overloaded'])
            is_timeout = any(x in err_low for x in ['timeout', 'timed out', 'deadline'])
            retryable  = is_rate or is_server or is_timeout

            if retryable and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                short_err = err_str[:250].replace('\n', ' ')
                tqdm.write(f'    [{type(e).__name__} attempt {attempt+1}/{max_retries}] '
                           f'sleeping {delay:.0f}s | {short_err}')
                time.sleep(delay)
            else:
                raise
    if last_err:
        raise last_err
    return None


def parse_judge_response(text: str) -> dict:
    """Extract JSON from judge response, tolerating markdown fences."""
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    try:
        obj = json.loads(text)
        labels = {}
        for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            val = obj.get(h, -1)
            labels[h] = int(bool(val)) if val in (0, 1, True, False) else -1
        labels['reasoning'] = str(obj.get('reasoning', ''))
        return labels
    except Exception:
        labels = {}
        for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            m = re.search(rf'"{h}"\s*:\s*([01])', text)
            labels[h] = int(m.group(1)) if m else -1
        labels['reasoning'] = ''
        return labels


def call_claude_judge(ground_truth: str, model_output: str) -> dict:
    """
    FIXED: removed temperature=0.0 because claude-opus-4-7 deprecated it
    and was returning HTTP 400 errors. Pattern matches the working
    CLAUDE_ReACT_Final_PARALLEL.ipynb generation code, which also omits
    temperature from the kwargs.
    """
    prompt = make_judge_prompt(ground_truth, model_output)
    def _call():
        resp = _anthropic_client.messages.create(
            model='claude-opus-4-7',           # Latest: Claude Opus 4.7
            max_tokens=1024,                   # Judge needs less than generation
            system=JUDGE_SYSTEM,
            messages=[{'role': 'user', 'content': prompt}],
            # NOTE: no `temperature` parameter — deprecated for this model.
        )
        return resp.content[0].text
    raw = _retry_call(_call)
    return parse_judge_response(raw)


def call_gpt_judge(ground_truth: str, model_output: str) -> dict:
    prompt = make_judge_prompt(ground_truth, model_output)
    def _call():
        resp = _openai_client.chat.completions.create(
            model='gpt-5.5',                   # Latest: GPT-5.5
            messages=[
                {'role': 'system', 'content': JUDGE_SYSTEM},
                {'role': 'user',   'content': prompt},
            ],
            max_completion_tokens=1024,        # gpt-5.5 requires max_completion_tokens (not max_tokens)
            # NOTE: gpt-5.5 only supports default temperature (1.0)
        )
        return resp.choices[0].message.content
    raw = _retry_call(_call)
    return parse_judge_response(raw)


def call_gemini_judge(ground_truth: str, model_output: str) -> dict:
    """
    FIXED: matches the working pattern from Gemini_LEAST-MOST_Final_PARALLEL.ipynb:
      - Explicit GenerateContentConfig (temperature, max_output_tokens, top_p, top_k)
      - On 429 RESOURCE_EXHAUSTED: wait 60 seconds (matches the working notebook's
        rate-limit handler) before raising back to _retry_call's exponential
        backoff. This is more aggressive than the generic _retry_call wait.
    """
    prompt = make_judge_prompt(ground_truth, model_output)
    full_prompt = JUDGE_SYSTEM + '\n\n' + prompt

    # Match the generation-side config from the working notebook
    _gemini_config = google_genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=1024,    # Judge needs less than generation's 4096
        top_p=0.8,
        top_k=10,
    )

    def _call():
        try:
            resp = _genai_client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=full_prompt,
                config=_gemini_config,
            )
            return resp.text
        except Exception as e:
            err = str(e)
            # On rate-limit, sleep an extra 60s before letting _retry_call
            # add its own exponential backoff. Matches working notebook L122-125.
            if '429' in err or 'RESOURCE_EXHAUSTED' in err or 'quota' in err.lower():
                tqdm.write(f'    [Gemini 429] sleeping 60s before retry...')
                time.sleep(60)
            raise

    raw = _retry_call(_call)
    return parse_judge_response(raw)


JUDGE_CALL_MAP = {
    'Claude': call_claude_judge,
    'GPT':    call_gpt_judge,
    'Gemini': call_gemini_judge,
}


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def load_checkpoint():
    if os.path.exists(PANEL_CHECKPOINT):
        with open(PANEL_CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt: dict):
    with open(PANEL_CHECKPOINT, 'w') as f:
        json.dump(ckpt, f)


# ------------------------------------------------------------------
# Run the panel
# ------------------------------------------------------------------

def run_judge_panel(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """
    For every row in df, call the cross-judges and store their labels.
    Returns a long-format DataFrame with one row per (triplet, judge).
    Fully checkpoint-resumable, parallelised with ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    ckpt = load_checkpoint()
    ckpt_lock = threading.Lock()
    rows_lock = threading.Lock()

    rows_out = []
    tasks = []
    cached_rows = []

    h_keys = ['H1','H2','H3','H4','H5','H6']

    for idx, row in df.iterrows():
        model      = row['model']
        technique  = row['technique']
        video      = row['video']
        crime_type = row['crime_type']
        gt         = str(row['ground_truth'])
        output     = str(row['model_output'])

        for judge in judges_for(model):
            ck_key = f'{idx}_{judge}'
            base_row = {
                'row_idx':    idx,
                'model':      model,
                'technique':  technique,
                'video':      video,
                'crime_type': crime_type,
                'judge':      judge,
            }
            if ck_key in ckpt:
                labels = ckpt[ck_key]
                cached_rows.append({
                    **base_row,
                    **{h: labels.get(h, -1) for h in h_keys},
                    'reasoning': labels.get('reasoning', ''),
                })
            else:
                tasks.append((idx, base_row, judge, gt, output, ck_key))

    n_judges_per_row = len(JUDGE_FILTER) if JUDGE_FILTER else 2
    print(f'\nPanel: {len(df):,} triplets × {n_judges_per_row} judge(s) = {len(tasks)+len(cached_rows):,} calls total')
    print(f'Already done (checkpoint): {len(cached_rows):,}')
    print(f'Remaining calls:           {len(tasks):,}')
    print(f'Workers:                   {max_workers}')

    rows_out.extend(cached_rows)

    if not tasks:
        print('Nothing to do — all calls already cached.')
        return pd.DataFrame(rows_out)

    pbar = tqdm(total=len(tasks), desc='Judge calls', unit='call', dynamic_ncols=True)

    def _process_task(task):
        idx, base_row, judge, gt, output, ck_key = task
        try:
            labels = JUDGE_CALL_MAP[judge](gt, output)
        except Exception as e:
            tqdm.write(f'  ERROR [{judge}] row {idx}: {str(e)[:200]}')
            labels = {h: -1 for h in h_keys}
            labels['reasoning'] = f'ERROR: {str(e)[:300]}'

        row_out = {
            **base_row,
            **{h: labels.get(h, -1) for h in h_keys},
            'reasoning': labels.get('reasoning', ''),
        }

        with ckpt_lock:
            ckpt[ck_key] = labels
            if len(ckpt) % 25 == 0:
                save_checkpoint(ckpt)

        with rows_lock:
            rows_out.append(row_out)

        return row_out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_task, t) for t in tasks]
        for _ in as_completed(futures):
            pbar.update(1)

    pbar.close()

    with ckpt_lock:
        save_checkpoint(ckpt)

    return pd.DataFrame(rows_out)


# ------------------------------------------------------------------
# Majority-vote aggregation
# ------------------------------------------------------------------

def majority_vote(vals):
    """Return 1 if majority of valid (0/1) values are 1, else 0. Ties → 1."""
    valid = [v for v in vals if v in (0, 1)]
    if not valid:
        return -1
    return 1 if sum(valid) >= len(valid) / 2 else 0


def aggregate_panel(df_raw: pd.DataFrame, df_triplets: pd.DataFrame) -> pd.DataFrame:
    h_cols = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']

    agg = (df_raw
           .groupby(['row_idx', 'model', 'technique', 'video', 'crime_type'])[h_cols]
           .agg(majority_vote)
           .reset_index())

    inv_map = {v: k for k, v in HTYPE_MAP.items()}
    for h in h_cols:
        agg[inv_map[h]] = agg[h]

    df_triplets_indexed = df_triplets.reset_index().rename(columns={'index': 'row_idx'})
    agg = agg.merge(
        df_triplets_indexed[['row_idx', 'ground_truth', 'model_output', 'model_output_full_len']],
        on='row_idx', how='left'
    )

    agg['hallucination_count'] = agg[h_cols].apply(
        lambda r: sum(v for v in r if v == 1), axis=1
    )
    agg['any_hallucination'] = (agg['hallucination_count'] > 0).astype(int)

    return agg


# ------------------------------------------------------------------
# Inter-judge agreement (Cohen's Kappa per judge pair)
# ------------------------------------------------------------------

def cohens_kappa(y1, y2):
    valid = [(a, b) for a, b in zip(y1, y2) if a in (0, 1) and b in (0, 1)]
    if len(valid) < 2:
        return float('nan')
    a = [v[0] for v in valid]
    b = [v[1] for v in valid]
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_a = (sum(a) / n) * (sum(b) / n) + ((n - sum(a)) / n) * ((n - sum(b)) / n)
    return (p_o - p_a) / (1 - p_a) if p_a < 1 else 1.0


def compute_inter_judge_agreement(df_raw: pd.DataFrame) -> dict:
    h_cols  = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
    judges  = df_raw['judge'].unique().tolist()
    pairs   = [(judges[i], judges[j])
               for i in range(len(judges)) for j in range(i + 1, len(judges))]
    results = {}
    for j1, j2 in pairs:
        pair_key = f'{j1}_vs_{j2}'
        results[pair_key] = {}
        df1 = df_raw[df_raw['judge'] == j1].set_index('row_idx')
        df2 = df_raw[df_raw['judge'] == j2].set_index('row_idx')
        common_idx = df1.index.intersection(df2.index)
        for h in h_cols:
            y1 = df1.loc[common_idx, h].tolist()
            y2 = df2.loc[common_idx, h].tolist()
            results[pair_key][h] = round(cohens_kappa(y1, y2), 4)
        y1_all = [v for h in h_cols for v in df1.loc[common_idx, h].tolist()]
        y2_all = [v for h in h_cols for v in df2.loc[common_idx, h].tolist()]
        results[pair_key]['overall'] = round(cohens_kappa(y1_all, y2_all), 4)
    return results


# ------------------------------------------------------------------
# EXECUTE PANEL
# ------------------------------------------------------------------

print('\n' + '=' * 70)
print('LLM Judge Panel')
print('=' * 70)

if os.path.exists(PANEL_RAW_PATH):
    df_panel_raw = pd.read_csv(PANEL_RAW_PATH)
    print(f'Loaded existing raw panel labels: {len(df_panel_raw):,} rows')
else:
    df_panel_raw = run_judge_panel(df_triplets, max_workers=4)
    df_panel_raw.to_csv(PANEL_RAW_PATH, index=False)
    print(f'\nSaved raw panel labels: {len(df_panel_raw):,} rows')

df_panel_raw.to_csv(JUDGE_LABELS_PATH, index=False)

print('\nComputing inter-judge agreement...')
agreement = compute_inter_judge_agreement(df_panel_raw)
with open(JUDGE_AGREEMENT_PATH, 'w') as f:
    json.dump(agreement, f, indent=2)

print('\nInter-judge agreement (Cohen\'s Kappa):')
for pair, kappas in agreement.items():
    print(f'  {pair}:')
    for h, k in kappas.items():
        print(f'    {h}: {k:.4f}')

print('\nAggregating votes...')
df_full = aggregate_panel(df_panel_raw, df_triplets)
df_full.to_csv(FULL_LABELED_PATH, index=False)
print(f'Saved full labeled dataset: {len(df_full):,} rows → {FULL_LABELED_PATH}')

print('\n' + '=' * 70)
print('Hallucination Summary')
print('=' * 70)
h_cols = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
h_names = {
    'H1': 'Scene Fabrication',
    'H2': 'Crime Misclassification',
    'H3': 'Crime Missed',
    'H4': 'Severity Minimization',
    'H5': 'Entity Fabrication',
    'H6': 'Phantom Actors',
}

valid_df = df_full[df_full[h_cols].apply(lambda r: all(v >= 0 for v in r), axis=1)]

print(f'\nHallucination rate by type (% of triplets):')
for h in h_cols:
    rate = valid_df[h].mean() * 100
    print(f'  {h} {h_names[h]:<26} {rate:5.1f}%')

print(f'\nOverall hallucination rate by model:')
print(valid_df.groupby('model')['any_hallucination'].mean().mul(100).round(1).to_string())

print(f'\nOverall hallucination rate by technique:')
print(valid_df.groupby('technique')['any_hallucination'].mean().mul(100).round(1).to_string())

print(f'\nHallucination rate by (model × technique):')
pivot = (valid_df.groupby(['model', 'technique'])['any_hallucination']
         .mean().mul(100).round(1).unstack(fill_value=0))
print(pivot)

print('\nDone. Outputs saved to:', OUTPUT_DIR)

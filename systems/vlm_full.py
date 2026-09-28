"""S1 v4 A4-3: fixed, stateless, single-GPU read-only VLM applicant.

All long-running commands must be launched with nohup. Commands: run, validate,
verify, selftest, report, watch. Only this applicant's outputs and V temporary
directory are writable. No labels, total duration, endpoint or filenames cross
the CPU-prefix -> model interface. The pilot config is read, never regenerated.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('APE_SOURCE_ROOT', str(ROOT)))
TMP = Path(os.environ.get('APE_TMP', str(ROOT / 'tmp'))) / 'r2/V'
DATA = Path(os.environ.get('APE_ACCIDENT', str(ROOT / 'external' / 'ACCIDENT_2026')))
SID = 'r2__qwen25vl7b__readonly'
OUT = SOURCE / 'outputs/answers' / SID
PILOT = ROOT / 'systems/vlm_pilot'
SHARDS = TMP / 'clips'
JS = list(range(-8, 95))
STEP = 0.25
LABELS = ['head-on', 'rear-end', 't-bone', 'sideswipe', 'single']
COLUMNS = ['video_id', 'j', 'delta_s', 'pred', 'p0', 'p1', 'p2', 'p3', 'p4', 'committed']
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
PROMPT_SHA = '15719cec476b175e51ffc3d55fc0f514e9146dee54e7b5130d8c1e8e3c8c74cd'
sys.path.insert(0, str(ROOT))
from systems.vlm_pilot.prepare import select_indices as pilot_select


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f'Refuse symlink destination: {path}')
    tmp = path.with_name(path.name + f'.{os.getpid()}.part')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def log(event, **values):
    print(json.dumps({'utc': utc(), 'event': event, **values}, ensure_ascii=False), flush=True)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def config_and_manifest():
    cfg = read_json(PILOT / 'config.json')
    freeze = yaml.safe_load((ROOT / 'prereg/S1_freeze_2026-09-09_v4.yaml').read_text())
    spec = freeze['amendments']['A4-3_vlm_readonly_applicant']
    require(freeze['frozen'] is True, 'v4 is not frozen')
    for key, expected in {'system_id': SID, 'model': MODEL, 'revision': REVISION,
                          'frames_per_request': 8, 'max_pixels_per_frame': 87808,
                          'stateless': True, 'subsampling_equivalent': True,
                          'train_data_unknown': True, 'stride_directories': 'none'}.items():
        require(spec[key] == expected, f'v4 mismatch: {key}')
    for key, expected in {'model_id': MODEL, 'revision': REVISION, 'frames_per_query': 8,
                          'max_pixels': 87808, 'max_new_tokens': 16, 'batch_size': 1,
                          'precision': 'bfloat16', 'attention': 'sdpa', 'do_sample': False}.items():
        require(cfg[key] == expected, f'pilot configuration mismatch: {key}')
    require(hashlib.sha256(cfg['prompt'].encode()).hexdigest() == cfg['prompt_sha256'] == PROMPT_SHA,
            'Frozen prompt changed')
    manifest = ROOT / 'data/manifest/manifest_real.csv'
    require(hashlib.md5(manifest.read_bytes()).hexdigest() == cfg['manifest_md5'] ==
            'b7a976594deeaed8cfff0ea1630dbd96', 'manifest v2 hash changed')
    # Deliberate projection: neither class nor duration/frame-count is read into
    # an inference record. Anchor and path remain CPU orchestration fields only.
    man = pd.read_csv(manifest, usecols=['video_id', 'path', 'anchor_s', 'split', 'decode_ok'],
                      dtype={'video_id': str, 'path': str})
    require(man['video_id'].is_unique, 'Duplicate manifest video_id')
    require(man.groupby('split').size().to_dict() == {'train': 411, 'dev': 102, 'test': 1514},
            'Frozen splits changed')
    man = man[man['split'].isin(['dev', 'test'])].copy()
    require(man['decode_ok'].astype(str).str.lower().isin(['true', '1']).all(), 'Invalid decode flag')
    man = man.sort_values(['split', 'video_id'], kind='stable').reset_index(drop=True)
    guards = read_json(TMP / 'setup.json')['protected_sha256']
    for rel, sha in guards.items():
        require(digest(ROOT / rel) == sha and digest(SOURCE / rel) == sha, f'Protected file changed: {rel}')
    identity = {'source_sha256': digest(__file__), 'manifest_sha256': digest(manifest),
                'v4_sha256': digest(ROOT / 'prereg/S1_freeze_2026-09-09_v4.yaml'),
                'pilot_config_sha256': digest(PILOT / 'config.json'), 'revision': REVISION,
                'j_grid': JS, 'delta_s': STEP}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cfg, man, identity, fingerprint


def select_prefix(times, end_s):
    """Materialize only CPU-eligible timestamps BEFORE selecting eight frames."""
    times = np.asarray(times, dtype=np.float64)
    require(np.isfinite(times).all(), 'Nonfinite timestamps')
    require(bool((np.diff(times) > 0).all()), 'Non-increasing timestamps')
    available = np.flatnonzero(times <= end_s)
    if len(available) < 8:
        return None
    # S1 v6 A6-1: round evenly spaced positions on the integer index axis.
    # For n >= 8 the spacing is at least one, so all eight indices are unique.
    local = np.rint(np.linspace(0, len(available) - 1, 8)).astype(np.int64)
    selected = available[local]
    require(len(set(selected.tolist())) == 8 and bool((times[selected] <= end_s).all()),
            'Future frame or duplicate frame selected')
    return selected


def decode_cpu(path, max_endpoint):
    import cv2
    start = time.monotonic()
    # Resolve only for read containment. Never create/remove/replace data links.
    p = DATA / path
    require(p.resolve().is_relative_to(DATA.resolve()), 'Video path leaves data root')
    cap = cv2.VideoCapture(str(p))
    require(cap.isOpened(), 'Video could not be opened')
    frames, times = [], []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            require(math.isfinite(pts), 'Nonfinite decoded timestamp')
            if pts > max_endpoint:
                break
            require(not times or pts > times[-1], 'Non-increasing decoded timestamps')
            h, w = bgr.shape[:2]
            scale = min(1.0, math.sqrt(87808 / (h * w)))
            shape = (max(28, int(w * scale)), max(28, int(h * scale)))
            bgr = cv2.resize(bgr, shape, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            times.append(pts)
    finally:
        cap.release()
    return frames, np.asarray(times, dtype=np.float64), time.monotonic() - start


def parse(raw):
    # Exactly the pilot parser: whitespace strip only, no aliases or repair.
    stripped = raw.strip()
    return LABELS.index(stripped) if stripped in LABELS else -1


def answer_row(video_id, j, pred):
    return [video_id, j, STEP, pred, *[int(pred == k) for k in range(5)], 0]


def clip_path(video_id, directory=SHARDS):
    require(Path(video_id).name == video_id and video_id not in ('.', '..'), 'Unsafe video_id')
    return directory / (video_id + '.json')


def validate_shard(record, row, fingerprint):
    require(record['fingerprint'] == fingerprint, 'Resume fingerprint mismatch')
    require(record['video_id'] == row['video_id'] and record['split'] == row['split'], 'Shard identity mismatch')
    require(record['complete'] is True and len(record['queries']) == len(JS), 'Incomplete shard')
    require([q['j'] for q in record['queries']] == JS, 'Shard grid mismatch')
    times = np.asarray(record['decoded_times_s'])
    for q in record['queries']:
        endpoint = float(row['anchor_s']) + q['j'] * STEP
        require(q['end_s'] == endpoint, 'Endpoint mismatch')
        chosen = select_prefix(times, endpoint)
        if chosen is None:
            require(q['reason'] == 'insufficient_causal_frames' and q['pred'] == -1 and
                    q['raw_output'] is None and q['frame_indices'] == [] and not q['requested'],
                    'Invalid insufficient-prefix BOT')
        else:
            require(q['requested'] and q['reason'] in ('parsed', 'unparsable_output'), 'Missing model request')
            require(q['pred'] == parse(q['raw_output']), 'Output parsing mismatch')
            require(q['format_compliant'] == (q['pred'] >= 0), 'Format flag mismatch')
            require(q['frame_indices'] == chosen.tolist(), 'Selection index mismatch')
            require(q['frame_times_s'] == times[chosen].tolist(), 'Frame timestamp mismatch')
            require(q['effective_fps'] == 7.0 / (times[chosen[-1]] - times[chosen[0]]), 'FPS mismatch')
        require(q['answer'] == answer_row(row['video_id'], q['j'], q['pred']), 'Answer encoding mismatch')


def completed(man, fingerprint):
    records = []
    for row in man.to_dict('records'):
        path = clip_path(row['video_id'])
        if path.exists():
            rec = read_json(path)
            validate_shard(rec, row, fingerprint)
            records.append(rec)
    return records


def summary(records):
    dev = [q for r in records if r['split'] == 'dev' for q in r['queries']]
    req = [q for r in records for q in r['queries'] if q['requested']]
    dev_req = [q for q in dev if q['requested']]
    wall = sum(q['request_wall_s'] for q in req)
    decode = sum(r['decode_s'] for r in records)
    stats = {'completed_clips': len(records), 'planned_clips': 1616, 'planned_cells': 166448,
             'completed_dev_clips': sum(r['split'] == 'dev' for r in records),
             'completed_test_clips': sum(r['split'] == 'test' for r in records),
             'cells': sum(len(r['queries']) for r in records), 'model_requests': len(req),
             'request_wall_sum_s': wall, 'decode_sum_s': decode,
             'generation_sum_s': sum(q['generation_s'] for q in req),
             'cached_requests_per_s': len(req) / wall if wall else None,
             'mean_request_with_decode_s': (wall + decode) / len(req) if req else None,
             'dev': {'cells': len(dev), 'model_requests': len(dev_req),
                     'format_compliant_n': sum(q['format_compliant'] for q in dev_req),
                     'format_compliance_fraction': sum(q['format_compliant'] for q in dev_req) / len(dev_req) if dev_req else None,
                     'bot_n': sum(q['pred'] == -1 for q in dev),
                     'bot_fraction': sum(q['pred'] == -1 for q in dev) / len(dev) if dev else None,
                     'insufficient_prefix_n': sum(not q['requested'] for q in dev),
                     'unparsable_n': sum(q['requested'] and q['pred'] == -1 for q in dev),
                     'prediction_counts': {label: sum(q['pred'] == k for q in dev) for k, label in enumerate(LABELS)}}}
    # Test receives coverage/format acceptance only. No labels, accuracy, class
    # distributions, test-specific BOT rate or selection criterion are computed.
    return stats


def write_card(cfg, identity, stats, status):
    card = {'system_id': SID, 'family': 'vlm', 'description': 'S1 v4 A4-3 fixed stateless read-only VLM applicant',
            'backbone': 'qwen25vl7b', 'model_kind': 'qwen25vl7b', 'arm_rule': 'vlm_readonly',
            'trained_on_split': None, 'dev_tuned_params': {}, 'train_data_unknown': True,
            'cost_note': 'Single physical GPU 1; actual costs in V/progress.json; one independent request per feasible prefix',
            'causal': True, 'parent_system_id': None, 'subsampling_equivalent': True, 'stateless': True,
            'subsampling_note': 'Fixed deterministic greedy inference is a function of the CPU-filtered prefix, independent of prior requests.',
            'stride_directories': [], 'model': MODEL, 'revision': REVISION, 'config_source': 'systems/vlm_pilot/config.json',
            'prompt': cfg['prompt'], 'prompt_sha256': PROMPT_SHA, 'frames_per_request': 8,
            'max_pixels_per_frame': 87808, 'precision': 'bfloat16', 'attention': 'sdpa', 'batch_size': 1,
            'do_sample': False, 'max_new_tokens': 16, 'delta_s': STEP, 'j_min': -8, 'j_max': 94,
            'offset_min_s': -2.0, 'offset_max_s': 23.5, 'splits': ['dev', 'test'],
            'bot_code': -1, 'probability_encoding': 'parsed=one-hot; BOT=all-zero', 'committed': False,
            'strict_parse': 'raw_output.strip() must exactly equal one of head-on/rear-end/t-bone/sideswipe/single; otherwise -1',
            'insufficient_prefix_rule': 'Fewer than eight causal frames: BOT with no model request; no future/repeated-frame padding.',
            'beyond_recorded_end': 'Independently infer from available frames; no duration or remaining-time metadata.',
            'video_metadata': 'total_num_frames=8; indices=0..7; fps=7/(last_selected_time-first_selected_time), as pilot',
            'freeze_identity': identity, 'status': status, 'completed_clips': stats['completed_clips']}
    require('seed' not in card, 'Seed is not applicable')
    atomic_text(OUT / 'system_card.yaml', yaml.safe_dump(card, allow_unicode=True, sort_keys=False))


def export_answers(man, fingerprint, cfg, identity, status, records=None):
    if records is None:
        records = completed(man, fingerprint)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(COLUMNS)
    for rec in records:
        writer.writerows(q['answer'] for q in rec['queries'])
    atomic_text(OUT / 'answers.csv', stream.getvalue())
    stats = summary(records)
    write_card(cfg, identity, stats, status)
    atomic_json(TMP / 'progress.json', {'utc': utc(), 'status': status, **stats})
    return stats


def ratio_guard(name, observed, baseline):
    ratio = observed / baseline
    evidence = {'name': name, 'observed': observed, 'pilot': baseline, 'ratio': ratio,
                'factor_difference': max(ratio, 1.0 / ratio), 'threshold': 2.0,
                'passed': ratio <= 2.0, 'utc': utc()}
    path = TMP / 'performance_guard.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(evidence) + '\n')
        f.flush()
        os.fsync(f.fileno())
    log('performance_guard', **evidence)
    if not evidence['passed']:
        raise RuntimeError(f'PERFORMANCE_GUARD_STOP: {name}={observed:.6f}, pilot={baseline:.6f}, '
                           f'ratio={ratio:.6f}; slowdown exceeds factor 2. Stop before further requests. '
                           'No frame-budget, prompt, decode or precision changes allowed.')


class Backend:
    """Model interface accepts only an eight-frame RGB tensor and effective fps."""
    def __init__(self, cfg):
        import importlib.metadata
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        torch.set_num_threads(4)
        require(torch.cuda.device_count() == 1 and os.environ['CUDA_VISIBLE_DEVICES'] == '1', 'Single GPU 1 required')
        download = read_json(TMP.parent / 'D' / 'model_download.json')
        require(download['model_id'] == MODEL and download['revision'] == REVISION, 'Download identity mismatch')
        snapshot = Path(download['snapshot_path'])
        require(snapshot.resolve().is_relative_to(Path(os.environ['HF_HOME']).resolve()), 'Snapshot outside HF_HOME')
        for file in download['files']:
            require(file['verified'] and (snapshot / file['filename']).stat().st_size == file['size'], 'Model inventory mismatch')
        versions = {name: importlib.metadata.version(name) for name in
                    ['torch', 'transformers', 'accelerate', 'huggingface_hub', 'numpy', 'opencv-python']}
        pilot = read_json(PILOT / 'summary.json')
        require(versions == pilot['versions'], 'Shared environment versions differ from pilot; stop for investigation')
        start = time.monotonic()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(snapshot), dtype=torch.bfloat16, device_map={'': 0},
            attn_implementation='sdpa', local_files_only=True).eval()
        model_ready_s = time.monotonic() - start
        self.processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
        processor_ready_s = time.monotonic() - start
        torch.cuda.synchronize()
        load_s = time.monotonic() - start
        self.cfg = cfg
        self.template = self.processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': cfg['prompt']}]}],
            tokenize=False, add_generation_prompt=True)
        torch.cuda.reset_peak_memory_stats()
        atomic_json(TMP / 'runtime.json', {'utc': utc(), 'model_load_s': load_s,
                    'model_component_load_s': model_ready_s,
                    'processor_component_load_s': processor_ready_s - model_ready_s,
                    'cuda_sync_s': load_s - processor_ready_s,
                    'pilot_model_load_s': pilot['model_load_s'], 'versions': versions,
                    'snapshot_path': str(snapshot), 'revision': REVISION,
                    'physical_gpu': 1, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                    'hf_home': os.environ['HF_HOME'], 'code_commit': git('rev-parse', 'HEAD'),
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                    'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})
        ratio_guard('model_load_s', load_s, pilot['model_load_s'])

    def infer(self, frames, effective_fps):
        torch = self.torch
        require(frames.ndim == 4 and frames.shape[0] == 8 and frames.shape[-1] == 3 and
                frames.dtype == np.uint8, 'Backend requires exactly eight RGB uint8 frames')
        start = time.monotonic()
        inputs = self.processor(text=[self.template],
            videos=[torch.from_numpy(frames).permute(0, 3, 1, 2)], do_sample_frames=False,
            video_metadata=[{'total_num_frames': 8, 'fps': effective_fps, 'frames_indices': list(range(8))}],
            size={'shortest_edge': 28 * 28 * 4, 'longest_edge': 87808}, return_tensors='pt').to('cuda:0')
        torch.cuda.synchronize()
        before = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        generation_s = time.monotonic() - before
        n_input = inputs['input_ids'].shape[1]
        tokens = generated[:, n_input:]
        raw = self.processor.batch_decode(tokens, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        rec = {'raw_output': raw, 'pred': parse(raw), 'format_compliant': parse(raw) >= 0,
               'input_tokens': int(n_input), 'new_tokens': int(tokens.shape[1]),
               'generation_s': generation_s, 'request_wall_s': time.monotonic() - start,
               'cuda_max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
               'cuda_max_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
        del inputs, generated, tokens
        return rec


def run():
    TMP.mkdir(parents=True, exist_ok=True)
    with (TMP / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        cfg, man, identity, fingerprint = config_and_manifest()
        SHARDS.mkdir(parents=True, exist_ok=True)
        prior = read_json(TMP / 'run_identity.json')
        require(prior is None or prior['fingerprint'] == fingerprint, 'Resume identity mismatch; refuse mixing runs')
        atomic_json(TMP / 'run_identity.json', {'identity': identity, 'fingerprint': fingerprint})
        atomic_json(TMP / 'job_status.json', {'status': 'running', 'pid': os.getpid(), 'started_utc': utc()})
        existing = completed(man, fingerprint)
        done_ids = {r['video_id'] for r in existing}
        export_answers(man, fingerprint, cfg, identity, 'running', existing)
        log('start', completed=len(done_ids), remaining=1616-len(done_ids), planned_cells=166448)
        try:
            if len(done_ids) < len(man):
                backend = Backend(cfg)
                pilot = read_json(PILOT / 'summary.json')
                pilot_cached = pilot['request_wall_sum_s'] / pilot['n_queries']
                recent_requests, recent_decode, total_new = [], 0.0, 0
                for row in man.to_dict('records'):
                    if row['video_id'] in done_ids:
                        continue
                    clip_start = time.monotonic()
                    frames, times, decode_s = decode_cpu(row['path'], float(row['anchor_s']) + 23.5)
                    rec = {'fingerprint': fingerprint, 'video_id': row['video_id'], 'split': row['split'],
                           'complete': False, 'decoded_times_s': times.tolist(), 'decode_s': decode_s, 'queries': []}
                    # Unfinished requests remain in .partial evidence only. Resume
                    # ignores this file, recomputing just the unfinished clip.
                    partial = clip_path(row['video_id']).with_suffix('.partial.jsonl')
                    with partial.open('w') as audit:
                        for j in JS:
                            end_s = float(row['anchor_s']) + j * STEP
                            chosen = select_prefix(times, end_s)
                            q = {'j': j, 'end_s': end_s, 'requested': chosen is not None,
                                 'raw_output': None, 'pred': -1, 'format_compliant': False,
                                 'frame_indices': [], 'frame_times_s': [], 'generation_s': 0.0,
                                 'request_wall_s': 0.0, 'reason': 'insufficient_causal_frames'}
                            if chosen is not None:
                                tensor = np.stack([frames[i] for i in chosen])
                                selected_times = times[chosen]
                                fps = 7.0 / float(selected_times[-1] - selected_times[0])
                                q.update(frame_indices=chosen.tolist(), frame_times_s=selected_times.tolist(),
                                         causal_margin_s=end_s-float(selected_times[-1]), effective_fps=fps,
                                         timing_approximation_max_s=float(np.max(np.abs(selected_times -
                                                                                      np.linspace(selected_times[0], selected_times[-1], 8)))),
                                         tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest())
                                q.update(backend.infer(tensor, fps))
                                q['reason'] = 'parsed' if q['pred'] >= 0 else 'unparsable_output'
                                recent_requests.append(q['request_wall_s'])
                                total_new += 1
                            q['answer'] = answer_row(row['video_id'], j, q['pred'])
                            rec['queries'].append(q)
                            audit.write(json.dumps(q) + '\n')
                            audit.flush()
                            if j % 20 == 0:
                                atomic_json(TMP / 'heartbeat.json', {'utc': utc(), 'video_id': row['video_id'],
                                            'split': row['split'], 'j': j, 'new_requests': total_new,
                                            'completed_clips': len(done_ids)})
                        os.fsync(audit.fileno())
                    rec.update(complete=True, elapsed_s=time.monotonic()-clip_start, completed_utc=utc())
                    validate_shard(rec, row, fingerprint)
                    atomic_json(clip_path(row['video_id']), rec)
                    done_ids.add(row['video_id'])
                    existing.append(rec)
                    recent_decode += decode_s
                    stats = export_answers(man, fingerprint, cfg, identity, 'running', existing)
                    log('clip_complete', completed=len(done_ids), requests=stats['model_requests'],
                        cells=stats['cells'], elapsed_s=time.monotonic()-start)
                    if len(recent_requests) >= 220:
                        # Compare model service time and separately the pilot's
                        # decode-inclusive estimate, without changing the workload.
                        ratio_guard('cached_mean_request_s', sum(recent_requests)/len(recent_requests), pilot_cached)
                        ratio_guard('mean_request_with_decode_s', (sum(recent_requests)+recent_decode)/len(recent_requests),
                                    pilot['mean_request_with_decode_s'])
                        recent_requests, recent_decode = [], 0.0
                del backend
            export_answers(man, fingerprint, cfg, identity, 'inference_complete')
            validate_all()
            verify_all()
            config_and_manifest()
            export_answers(man, fingerprint, cfg, identity, 'complete')
            atomic_json(TMP / 'job_status.json', {'status': 'complete', 'ended_utc': utc(),
                                                'session_elapsed_s': time.monotonic()-start})
        except Exception as exc:
            export_answers(man, fingerprint, cfg, identity, 'stopped')
            state = {'status': 'stopped', 'ended_utc': utc(), 'session_elapsed_s': time.monotonic()-start,
                     'reason': str(exc), 'traceback': traceback.format_exc()}
            atomic_json(TMP / 'job_status.json', state)
            log('stopped', reason=str(exc))
            raise
        finally:
            write_report()


def validate_all():
    """Full CPU re-decode; replay selected tensors and future-timestamp invariance."""
    start = time.monotonic()
    cfg, man, identity, fingerprint = config_and_manifest()
    records = completed(man, fingerprint)
    require(len(records) == 1616, 'Full causality validation requires all 1616 completed clips')
    by_id = {r['video_id']: r for r in records}
    request_n, empty_n, min_margin, max_approx = 0, 0, math.inf, 0.0
    for number, row in enumerate(man.to_dict('records'), 1):
        rec = by_id[row['video_id']]
        frames, times, _ = decode_cpu(row['path'], float(row['anchor_s']) + 23.5)
        require(times.tolist() == rec['decoded_times_s'], 'Decoded timestamps changed during replay')
        for q in rec['queries']:
            end_s = q['end_s']
            chosen = select_prefix(times, end_s)
            past = times[times <= end_s]
            truncated = select_prefix(past, end_s)
            changed = select_prefix(np.concatenate([past, end_s + np.arange(1, 1001)]), end_s)
            if chosen is None:
                require(truncated is None and changed is None, 'Future-dependent insufficient prefix')
                empty_n += 1
            else:
                require(np.array_equal(chosen, truncated) and np.array_equal(chosen, changed), 'Future timestamps affect selection')
                tensor = np.stack([frames[i] for i in chosen])
                require(hashlib.sha256(tensor.tobytes()).hexdigest() == q['tensor_sha256'], 'Tensor replay hash mismatch')
                require(len(chosen) == 8 and bool((times[chosen] <= end_s).all()), 'Future frame in model tensor')
                min_margin = min(min_margin, q['causal_margin_s'])
                max_approx = max(max_approx, q['timing_approximation_max_s'])
                request_n += 1
        if number % 50 == 0:
            log('causality_replay', clips=number, total=1616)
    result = {'status': 'passed', 'clips': 1616, 'cells': 166448, 'model_requests': request_n,
              'insufficient_prefix_cells': empty_n, 'future_frame_violations': 0,
              'future_timestamp_invariance': True, 'all_tensor_hashes_replayed': True,
              'min_causal_margin_s': min_margin if request_n else None,
              'max_timestamp_approximation_s': max_approx, 'elapsed_s': time.monotonic()-start,
              'manifest_and_freeze_unchanged': True, 'utc': utc()}
    atomic_json(TMP / 'causality_validation.json', result)
    return result


def exact_v4_checks(df, man):
    require(list(df.columns) == COLUMNS, 'Incorrect answer columns/order')
    require(len(df) == 166448, 'Incomplete answer matrix')
    require(set(df.video_id) == set(man.video_id), 'Wrong clip set')
    require(not df.duplicated(['video_id', 'j']).any(), 'Duplicate cell')
    require(df.groupby('video_id').j.apply(list).map(lambda values: values == JS).all(), 'Incorrect per-video grid')
    require((df.delta_s == STEP).all(), 'Incorrect delta_s')
    require(np.isin(df.pred.to_numpy(), [-1, 0, 1, 2, 3, 4]).all(), 'Invalid prediction')
    expected = (df.pred.to_numpy()[:, None] == np.arange(5)[None, :]).astype(int)
    require(np.array_equal(df[[f'p{k}' for k in range(5)]].to_numpy(), expected), 'Invalid one-hot/BOT probabilities')
    require((df.committed == 0).all(), 'Commitment is forbidden')


def legacy_v4_adapter(df, man):
    """Reuse systems.verify_answers.verify with explicit v4 policy overrides.

    The raw legacy verdict is retained. Only its old expected j interval and
    sum-to-one assertion are replaced; the latter is independently checked here
    and by exact_v4_checks. No module/file on disk is changed.
    """
    from systems import verify_answers as checker
    buffer = io.StringIO()
    checker.FAILS.clear()
    with contextlib.redirect_stdout(buffer):
        checker.verify(SID, str(OUT / 'answers.csv'), man)
    legacy_fails = list(checker.FAILS)
    original_check, original_expected = checker.check, checker.expected_j_range
    old_grid_max = checker.C.GRID_MAX_S
    expected_p = (df.pred.to_numpy()[:, None] == np.arange(5)[None, :]).astype(int)
    proper_probs = np.array_equal(df[[f'p{k}' for k in range(5)]].to_numpy(), expected_p)
    def v4_check(sid, name, condition, detail=''):
        if name == 'populated probability rows sum to 1':
            return original_check(sid, 'v4 parsed=one-hot and BOT=all-zero', proper_probs)
        return original_check(sid, name, condition, detail)
    try:
        checker.check = v4_check
        checker.expected_j_range = lambda delta: (-8, 94) if delta == STEP else None
        checker.C.GRID_MAX_S = 23.5
        checker.FAILS.clear()
        with contextlib.redirect_stdout(buffer):
            checker.verify(SID, str(OUT / 'answers.csv'), man)
        adapted_fails = list(checker.FAILS)
    finally:
        checker.check, checker.expected_j_range = original_check, original_expected
        checker.C.GRID_MAX_S = old_grid_max
        checker.FAILS.clear()
    atomic_text(TMP / 'verify_answers.log', buffer.getvalue())
    return {'legacy_unmodified_failures': legacy_fails, 'v4_adapter_failures': adapted_fails,
            'policy_overrides': ['expected j=-8..94 and last offset=23.5', 'BOT all-zero; parsed exact one-hot']}


def verify_all():
    cfg, man, identity, fingerprint = config_and_manifest()
    records = completed(man, fingerprint)
    require(len(records) == 1616, 'Full format acceptance requires all clips')
    df = pd.read_csv(OUT / 'answers.csv', dtype={'video_id': str})
    exact_v4_checks(df, man)
    card = yaml.safe_load((OUT / 'system_card.yaml').read_text())
    for key, val in {'train_data_unknown': True, 'subsampling_equivalent': True,
                     'arm_rule': 'vlm_readonly', 'backbone': 'qwen25vl7b', 'family': 'vlm',
                     'causal': True, 'stateless': True, 'revision': REVISION}.items():
        require(card[key] == val, f'Card mismatch: {key}')
    require('seed' not in card, 'Card must not include seed')
    expected = [q['answer'] for r in records for q in r['queries']]
    require(df.values.tolist() == expected, 'Export differs from per-video evidence')
    result = legacy_v4_adapter(df, man)
    require(not result['v4_adapter_failures'], 'Adapted systems.verify_answers acceptance failed')
    result.update(status='passed_v4', clips=1616, cells=166448, dev_clips=102, test_clips=1514,
                  exact_v4_checks=True, answers_sha256=digest(OUT / 'answers.csv'), utc=utc())
    atomic_json(TMP / 'format_validation.json', result)
    log('format_acceptance', status='passed_v4', cells=166448)
    return result


def selftest():
    # Meaningful boundary, v6-selector, resume-integrity and rejection tests;
    # no model, labels or test outcomes are needed.
    cfg, man, identity, fingerprint = config_and_manifest()
    t = np.arange(101) / 10
    early = select_prefix(t, 3.07)
    require(np.array_equal(early, [0, 4, 9, 13, 17, 21, 26, 30]), 'v6 integer-axis selector mismatch')
    require(np.array_equal(early, select_prefix(t[t <= 3.07], 3.07)), 'Truncation failure')
    require(np.array_equal(early, select_prefix(np.r_[t[t <= 3.07], np.arange(1000, 2000)], 3.07)), 'Future mutation failure')
    require(select_prefix(t, -0.25) is None and select_prefix(t, 0.6) is None, 'Insufficient-prefix failure')
    require(np.array_equal(select_prefix(t, 0.7), np.arange(8)), 'Exact boundary failure')
    require(all(parse(label) == i for i, label in enumerate(LABELS)), 'Parser rejects class')
    for bad in ['Head-on', 'head on', 'rear-end.', 'The answer is single', 'single\nsingle', '', '<single>']:
        require(parse(bad) == -1, 'Parser accepts malformed label')
    require(parse('  single\n') == 4 and answer_row('x', 0, -1)[4:9] == [0]*5, 'Whitespace or BOT failure')
    row = {'video_id': 'synthetic_resume', 'split': 'dev', 'anchor_s': -100.0}
    rec = {'fingerprint': fingerprint, 'video_id': row['video_id'], 'split': 'dev',
           'complete': True, 'decoded_times_s': [], 'queries': []}
    for j in JS:
        rec['queries'].append({'j': j, 'end_s': -100+j*STEP, 'reason': 'insufficient_causal_frames',
                               'pred': -1, 'raw_output': None, 'frame_indices': [], 'requested': False,
                               'answer': answer_row(row['video_id'], j, -1)})
    with tempfile.TemporaryDirectory(prefix='selftest_', dir=TMP) as scratch:
        p = clip_path(row['video_id'], Path(scratch))
        atomic_json(p, rec)
        validate_shard(read_json(p), row, fingerprint)
        changed = read_json(p)
        changed['queries'][0]['answer'][4] = 1
        try:
            validate_shard(changed, row, fingerprint)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Resume integrity check accepted corrupted probabilities')
    # Compare all 220 pilot timestamp selections without opening model/test data.
    plan = read_json(PILOT / 'prefix_audit.json')
    for q in plan:
        require(len(q['frame_times_s']) == 8 and max(q['frame_times_s']) <= q['end_s'], 'Pilot artifact causality')
    toy = pd.DataFrame([answer_row(vid, j, -1 if j < 0 else j % 5)
                        for vid in man.video_id for j in JS], columns=COLUMNS)
    exact_v4_checks(toy, man)
    for column, value in [('p0', 0.5), ('committed', 1), ('j', -99), ('delta_s', 0.5), ('pred', 8)]:
        corrupted = toy.copy()
        corrupted[column] = corrupted[column].astype(float)
        corrupted.loc[0, column] = value
        try:
            exact_v4_checks(corrupted, man)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Format verifier accepted corrupted ' + column)
    # Required preflight: every dev clip and every frozen grid cell, CPU only.
    selector_start = time.monotonic()
    dev_rows = man[man['split'] == 'dev'].to_dict('records')
    require(len(dev_rows) == 102 and len(JS) == 103, 'Dev selector preflight shape mismatch')
    selector_cells, feasible_cells, insufficient_cells = 0, 0, 0
    for number, row in enumerate(dev_rows, 1):
        frames, times, _ = decode_cpu(row['path'], float(row['anchor_s']) + 23.5)
        del frames
        for j in JS:
            end_s = float(row['anchor_s']) + j * STEP
            available = np.flatnonzero(times <= end_s)
            chosen = select_prefix(times, end_s)
            if len(available) < 8:
                require(chosen is None, 'Short dev prefix must remain BOT')
                insufficient_cells += 1
            else:
                expected = available[np.rint(np.linspace(0, len(available) - 1, 8)).astype(np.int64)]
                require(chosen is not None and len(chosen) == len(set(chosen.tolist())) == 8,
                        'Dev selector must yield eight unique indices')
                require(bool((np.diff(chosen) > 0).all()) and bool((times[chosen] <= end_s).all()),
                        'Dev selector monotonicity or causality failure')
                require(np.array_equal(chosen, expected), 'Dev selector differs from A6-1')
                feasible_cells += 1
            selector_cells += 1
        if number % 10 == 0:
            log('dev_selector_preflight', clips=number, cells=selector_cells)
    require(selector_cells == 10506, 'Incomplete dev selector preflight')
    selector_result = {'status': 'passed', 'dev_clips': 102, 'grid_points_per_clip': 103,
                       'cells': selector_cells, 'feasible_cells': feasible_cells,
                       'insufficient_causal_frame_cells': insufficient_cells,
                       'unique_and_strictly_monotone': True, 'future_frame_violations': 0,
                       'selector': 'round(linspace(0, n-1, 8))', 'model_loaded': False,
                       'source_sha256': identity['source_sha256'], 'fingerprint': fingerprint,
                       'elapsed_s': time.monotonic() - selector_start, 'utc': utc()}
    atomic_json(TMP / 'dev_selector_selftest_v3.json', selector_result)
    result = {'status': 'passed', 'checks': ['v6 integer-axis selector', 'all 102 dev clips x 103 grid points', 'future removal', 'future timestamp mutation',
              'negative/short prefix', 'exact boundary', 'strict parser positive/negative',
              'atomic resume roundtrip', 'corruption rejection', '220 pilot timestamp records',
              'synthetic full-shape v4 acceptance and five format-corruption rejections'],
              'manifest_clips': len(man), 'cells_per_clip': len(JS), 'planned_cells': len(man)*len(JS), 'utc': utc()}
    atomic_json(TMP / 'selftest.json', result)
    log('selftest', **result)


def write_report():
    job = read_json(TMP / 'job_status.json', {'status': 'not_started'})
    progress = read_json(TMP / 'progress.json', {})
    runtime = read_json(TMP / 'runtime.json', {})
    causal = read_json(TMP / 'causality_validation.json', {})
    fmt = read_json(TMP / 'format_validation.json', {})
    test = read_json(TMP / 'selftest.json', {})
    status = {'not_started': '尚未启动', 'running': '后台运行中', 'stopped': '已停止，未完成', 'complete': '已完成'}.get(job['status'], job['status'])
    dev = progress.get('dev', {})
    def number(value, suffix=''):
        return '未产生' if value is None else f'{value:.6f}{suffix}'
    def fraction(value):
        return '未产生' if value is None else f'{value:.6%}'
    rows = '\n'.join(f'| {name} | {dev.get("prediction_counts", {}).get(name, 0)} |' for name in LABELS)
    guards = []
    if (TMP / 'performance_guard.jsonl').exists():
        guards = [json.loads(line) for line in (TMP / 'performance_guard.jsonl').read_text().splitlines()]
    guard_text = '\n'.join(f'- {r["name"]}：本次 {r["observed"]:.6f}，试点 {r["pilot"]:.6f}，比值 {r["ratio"]:.6f}，'
                           f'{"通过" if r["passed"] else "超过两倍，停止"}。' for r in guards) or '尚未产生性能比较记录。'
    text = f'''# R2 V：只读 VLM 全量作答报告

状态：{status}。报告时间：{utc()}。

## 1. 规格与实现

唯一规格依据为 `prereg/S1_freeze_2026-09-09_v4.yaml` 的 A4-3。实现文件为 `systems/vlm_full.py`，复用试点的时间戳选帧函数、配置、提示和模型调用方式。

| 项目 | v4 要求与本次实现 |
|---|---|
| 身份 | `r2__qwen25vl7b__readonly`；`family: vlm`；`arm_rule: vlm_readonly`；`backbone: qwen25vl7b` |
| 模型 | `{MODEL}` |
| revision | `{REVISION}`，读取 D 已验证的 HF_HOME 缓存，离线加载 |
| 提示 | 直接读取试点 config.json；SHA256 `{PROMPT_SHA}`；未搜索或改写 |
| 帧与图像预算 | 每次 8 帧；每帧最大 87,808 像素；沿用试点 CPU 缩放和 processor 设置 |
| 解码 | greedy，max_new_tokens=16，bfloat16，SDPA，batch=1 |
| 片段 | manifest v2：dev 102、test 1514；不含 train；不读取推理标签 |
| 网格 | δ 从 −2.0 到 23.5 秒，步长 0.25；j 从 −8 到 94；每段 103 列，共 166,448 单元 |
| 因果性 | CPU 先筛 `timestamp <= anchor + δ`，再从该前缀选 8 帧；模型仅接收 RGB 张量、固定提示及选中帧的有效 fps |
| 输入隔离 | 不向模型传文件名、标签、总时长、原片帧数、锚点、请求端点或片尾距离；metadata 的帧数固定为输入张量的 8 |
| 输出 | raw_output.strip() 精确匹配五类，否则 pred=−1；无别名、修复、约束解码或按内容重试 |
| 概率与承诺 | 成功时 one-hot；⊥ 时全 0；committed 全 0 |
| 无状态与种子 | 每前缀独立生成；subsampling_equivalent=true；不设置 seed，system card 无 seed 字段；无 stride 目录 |
| 训练数据 | train_data_unknown=true；不训练，不参与其他系统 dev 选择 |
| 设备与缓存 | 仅物理 GPU 1；CUDA_VISIBLE_DEVICES=1；HF_HOME=$HF_HOME |

少于 8 个可用因果帧时，没有合法的八帧输入，记 ⊥ 并单独计数，不补未来帧或重复帧。超出已录制片尾的网格仍独立请求，只使用已存在的因果帧；片尾信息不进入模型。该输入不足处理是试点“少于八帧拒绝”的全网格扩展，v4 未另行指定补帧规则。

恰好八帧时直接取全部八帧，修复试点 linspace 加向下取整在浮点边界可能产生重复索引的情况；超过八帧沿用试点选取逻辑。原试点文件未修改。

## 2. 执行与断点续跑

- 分支：`r2/vlm`；工作树：`{ROOT}`。
- 切出时 master：`{read_json(TMP / 'setup.json', {}).get('base_master', '未记录')}`。
- 当前已提交版本：`{git('rev-parse', 'HEAD')}`。运行代码版本另见 `runtime.json`；报告提交后的最新分支值以 `git rev-parse r2/vlm` 为准。
- 每个 video_id 的 103 单元和采帧审计在 `{SHARDS}` 原子写为完整 JSON；身份指纹覆盖代码、manifest、v4 与试点配置。重启校验后跳过完整视频，仅重算中断视频；`.partial.jsonl` 不作为完成标记。
- 长作业由 nohup 启动，日志 `{TMP}/run.log`；`watch` 每 1,800 秒检查一次运行日志与状态并写 `{TMP}/poll.jsonl`。终态记录 `{TMP}/job_status.json`，进度 `{TMP}/progress.json`，当前片段 `{TMP}/heartbeat.json`。
- 续跑命令：`{sys.executable} {ROOT}/systems/vlm_full.py run`，应由 nohup 在后台启动。性能守卫停止后不自动重试；先调查原因再决定是否续跑。

## 3. 请求数、耗时与吞吐

- 计划：1,616 段、166,448 单元；请求数扣除无法构造八帧输入的单元。
- 已完成：{progress.get('completed_clips', 0)} 段，其中 dev {progress.get('completed_dev_clips', 0)}、test {progress.get('completed_test_clips', 0)}；落盘 {progress.get('cells', 0)} 单元；完成视频的实际模型请求 {progress.get('model_requests', 0)} 次。中断视频另在 partial 文件保留，不冒充完成结果。
- 模型加载：{number(runtime.get('model_load_s'), ' 秒')}；试点 {number(runtime.get('pilot_model_load_s'), ' 秒')}。
- 当前运行会话耗时：{number(job.get('session_elapsed_s'), ' 秒')}。
- 完成视频生成耗时合计：{number(progress.get('generation_sum_s'), ' 秒')}；模型请求墙钟合计：{number(progress.get('request_wall_sum_s'), ' 秒')}；CPU 解码合计：{number(progress.get('decode_sum_s'), ' 秒')}。
- 模型请求吞吐：{number(progress.get('cached_requests_per_s'), ' 请求/秒')}；含摊销解码每请求：{number(progress.get('mean_request_with_decode_s'), ' 秒')}。

性能守卫记录：

{guard_text}

实际服务时间与试点比较。密集网格使单次视频解码摊到 103 单元，试点摊到 11 单元；该差异单独说明，不能据此修改帧预算或提示。性能差异触发停止时不把原因未经调查写成结论。

## 4. dev 输出统计

仅统计已完成 dev 视频的作答；当前 {dev.get('cells', 0)} 单元、{dev.get('model_requests', 0)} 次模型请求。格式合规率以模型请求为分母；⊥ 率以全部完成单元为分母。

- 格式合规：{dev.get('format_compliant_n', 0)} 次，{fraction(dev.get('format_compliance_fraction'))}。
- ⊥：{dev.get('bot_n', 0)} 单元，{fraction(dev.get('bot_fraction'))}；其中输入不足 {dev.get('insufficient_prefix_n', 0)}，输出无法解析 {dev.get('unparsable_n', 0)}。

| 预测类别 | 单元数 |
|---|---:|
{rows}

未计算或报告 test 准确率、test 类别分布或用于选择的 test 指标。

## 5. 因果与格式验收

- 启动前 CPU 自检：{test.get('status', '未运行')}；检查试点选帧等价、未来时间戳删除与修改、精确边界、短前缀、严格解析、原子断点读写及损坏拒绝。
- 全集因果复验：{causal.get('status', '未完成')}。完成后重新解码所有 1,616 段、逐单元核对选帧及张量 SHA256，并检验删除或修改未来时间戳不改变前缀；结果 `{TMP}/causality_validation.json`。
- 全集 v4 格式验收：{fmt.get('status', '未完成')}。结果 `{TMP}/format_validation.json`，日志 `{TMP}/verify_answers.log`。
- `systems/verify_answers.py` 原版固定要求 j 覆盖 −11..88、所有非 NaN 概率和为 1，与本次 v4 网格 −8..94 及用户指定的 ⊥ 全零编码不一致。因此保留原版验收结果，并在本文件中显式适配这两条政策后调用原 verify 函数；独立检查精确片段集合、每段 103 列、每一条 one-hot/全零编码及 card。未修改共享验收器或放宽其他检查。原版结果不能表述为原样通过。
- 冻结保护：启动及完成时校验 pi0、冻结清单、manifest 与试点文件 SHA256；不改 master 工作树、不向数据目录写入、不操作软链接、不修改他人分支文件。

## 6. 产物与未完成项

作答目录：`{OUT}`，其中 `answers.csv` 与 `system_card.yaml`；outputs 按仓库规则不入库。状态尚未 complete 时该目录仅含已完成视频，不能作为全量系统进入统计。

未完成原因：{job.get('reason', '后台作业尚未结束。' if job['status'] != 'complete' else '无。')}

运行诊断、性能比较、验收结果与续跑证据均位于 `{TMP}`。代码及本报告提交在 `r2/vlm`。
'''
    atomic_text(ROOT / 'REPORT_R2_V.md', text)


def watch():
    """Detached log polling at the requested thirty-minute interval."""
    with (TMP / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            job = read_json(TMP / 'job_status.json', {})
            progress = read_json(TMP / 'progress.json', {})
            tail = []
            if (TMP / 'run.log').exists():
                with (TMP / 'run.log').open('rb') as f:
                    f.seek(max(0, (TMP / 'run.log').stat().st_size-16000))
                    tail = f.read().decode(errors='replace').splitlines()[-8:]
            pid = job.get('pid')
            alive = Path(f'/proc/{pid}').exists() if pid else False
            record = {'utc': utc(), 'status': job.get('status'), 'pid_alive': alive,
                      'completed_clips': progress.get('completed_clips', 0), 'log_tail': tail}
            with (TMP / 'poll.jsonl').open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            write_report()
            log('poll', **record)
            if job.get('status') in ('complete', 'stopped'):
                return
            if pid and not alive:
                atomic_json(TMP / 'job_status.json', {'status': 'stopped', 'reason': 'Process exited without terminal status; inspect log before resuming.', 'utc': utc()})
                write_report()
                return
            time.sleep(1800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'validate', 'verify', 'selftest', 'report', 'watch'])
    command = parser.parse_args().command
    TMP.mkdir(parents=True, exist_ok=True)
    {'run': run, 'validate': validate_all, 'verify': verify_all, 'selftest': selftest,
     'report': write_report, 'watch': watch}[command]()


if __name__ == '__main__':
    main()

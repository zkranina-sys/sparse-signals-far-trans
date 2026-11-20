"""Download only public pinned weights; no dataset is read by this script."""
import json
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from huggingface_hub import snapshot_download

t = time.perf_counter()
snapshot = snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct',
    revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
    allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'vocab.json', 'merges.txt', 'LICENSE', 'README.md'],
    max_workers=3,
)
result = {'model': 'Qwen/Qwen2.5-3B-Instruct',
          'revision': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
          'snapshot_path': snapshot, 'download_seconds': time.perf_counter()-t,
          'data_read': False}
(ROOT/'evidence/model_download.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))

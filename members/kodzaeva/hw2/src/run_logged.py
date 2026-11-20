"""Run a real local command and preserve its exact output and timing."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
if not cmd:
    p.error('command required')
root = pathlib.Path(__file__).resolve().parents[1]
logs = root / 'logs'
logs.mkdir(exist_ok=True)
receipt = root / 'evidence' / f'{a.label}_execution.json'
if receipt.exists():
    raise SystemExit('Refusing to overwrite an existing execution receipt')
env = dict(os.environ, PYTHONHASHSEED='0', HF_HUB_OFFLINE='1',
           TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
           HF_HOME=str(root/'.cache/huggingface'), TOKENIZERS_PARALLELISM='false')
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
t = time.perf_counter()
with (logs/f'{a.label}_stdout.txt').open('w') as out, (logs/f'{a.label}_stderr.txt').open('w') as err:
    run = subprocess.run(cmd, cwd=root, env=env, stdout=out, stderr=err)
record = {'command':cmd, 'started_at_utc':started,
          'ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'whole_process_seconds':time.perf_counter()-t, 'exit_code':run.returncode,
          'stdout':f'logs/{a.label}_stdout.txt','stderr':f'logs/{a.label}_stderr.txt',
          'offline_environment':{k:env[k] for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','PYTHONHASHSEED']}}
receipt.write_text(json.dumps(record,indent=2))
print(json.dumps(record))
raise SystemExit(run.returncode)

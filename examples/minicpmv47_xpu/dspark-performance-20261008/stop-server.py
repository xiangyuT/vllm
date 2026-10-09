import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
label=sys.argv[1]
receipt=json.loads((root/f'{label}-launch.json').read_text())
container=receipt['container']
code='''import os,signal,time
from pathlib import Path
for path in Path("/proc").iterdir():
 if not path.name.isdigit(): continue
 try: args=(path/"cmdline").read_bytes().split(b"\\0")
 except (OSError,ProcessLookupError): continue
 if b"vllm.entrypoints.cli.main" in args and b"31065" in args:
  os.kill(int(path.name),signal.SIGTERM)
  print("Stopped owned server",path.name)
time.sleep(5)
'''
result=subprocess.run(['docker','exec',container,'/workspace/runtime/.venv/bin/python','-c',code],
                      check=True,capture_output=True,text=True)
(root/f'{label}-shutdown.txt').write_text(result.stdout+result.stderr)
subprocess.run(['docker','stop','--time','5',container],check=True)
for card in [0,1]:
    result=subprocess.run(['xpu-smi','ps','-d',str(card),'-j'],check=True,capture_output=True,text=True)
    (root/f'{label}-gpu{card}-after.json').write_text(result.stdout)
    data=json.loads(result.stdout)
    assert all(row['process_name']=='xpu-smi' for row in data['device_util_by_proc_list']),data
print('Stopped stage and checked GPU 0/1 idle',label,flush=True)

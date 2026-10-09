import json
from pathlib import Path
import sys
import urllib.request

root=Path(__file__).resolve().parent
label=sys.argv[1]
base='http://127.0.0.1:31065'
prompt=next(row['text'] for row in json.loads(
    (root.parent/'dspark-gpu01-20261008/correctness-cases.json').read_text())
    if row.get('prompt_tokens')==2048)

def post(path,body):
    request=urllib.request.Request(base+path,data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(request,timeout=300)

with post('/start_profile',{'profile_prefix':label+'_decode','delay_iterations':8,'max_iterations':4}) as response:
    assert response.status==200
with post('/v1/chat/completions',{'model':'MiniCPM-V-4.7-1008',
          'messages':[{'role':'user','content':prompt}],
          'temperature':0,'seed':7,'max_tokens':32,'min_tokens':32,'ignore_eos':True,
          'chat_template_kwargs':{'enable_thinking':False}}) as response:
    data=json.load(response)
assert data['usage']['prompt_tokens']==2048 and data['usage']['completion_tokens']==32
(root/f'{label}-profile-request.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with post('/stop_profile',{}) as response:
    assert response.status==200
print('Profile completed outside timing samples',label,flush=True)

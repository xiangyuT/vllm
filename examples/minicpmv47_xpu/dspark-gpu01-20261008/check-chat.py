import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

root = Path(__file__).resolve().parent
profile = sys.argv[1]
base = 'http://127.0.0.1:31065'
model = 'MiniCPM-V-4.7-1008'

def post(endpoint, body, timeout=300):
    request = urllib.request.Request(base + endpoint, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.load(response)

deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(base + '/health', timeout=2) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(3)
else:
    raise RuntimeError('Service readiness timeout')
print(profile, 'ready', flush=True)

cases = [case for case in json.loads(
    (root.parent/'graph-experiment-20261008/correctness-cases.json').read_text())
    if case['case'] != 'context_32768_decode64']
(root/'correctness-cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
reference_path = root/'baseline_eager-correctness.json'
reference = None if profile == 'baseline_eager' else json.loads(reference_path.read_text())
results = []
for index, case in enumerate(cases):
    body = {'model':model, 'messages':[{'role':'user','content':case['text']}],
            'temperature':0,'seed':7,'max_tokens':64,'return_token_ids':True,
            'chat_template_kwargs':{'enable_thinking':False}}
    if case['force64']:
        body.update(min_tokens=64,ignore_eos=True)
    try:
        status,response = post('/v1/chat/completions',body)
    except urllib.error.HTTPError as error:
        (root/f'{profile}-request-error.json').write_text(json.dumps({
            'case':case['case'],'status':error.code,'body':error.read().decode()},indent=2)+'\n')
        raise
    choice=response['choices'][0]
    text=choice['message']['content'] or ''
    ids=choice['token_ids']
    assert len(ids)==response['usage']['completion_tokens']
    expected_ok=case['expected'] is None or case['expected'].lower() in text.lower()
    if case['force64']:
        assert response['usage']['prompt_tokens']==case['prompt_tokens']
        assert response['usage']['completion_tokens']==64
    matches=reference is None or ids==reference[index]['token_ids']
    result={'case':case['case'],'http_status':status,'response':response,
            'token_ids':ids,'expected_ok':expected_ok,
            'matches_baseline':None if reference is None else matches,
            'passed':status==200 and expected_ok and matches}
    if not matches:
        ref_ids=reference[index]['token_ids']
        result['first_difference']=next((i for i,(a,b) in enumerate(zip(ids,ref_ids)) if a!=b),min(len(ids),len(ref_ids)))
    results.append(result)
    (root/f'{profile}-correctness.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(profile,case['case'],'PASS' if result['passed'] else 'DIFF',repr(text[:80]),flush=True)
assert all(row['passed'] for row in results),'Functional parity failed; review before performance claims'
print(profile,f'COMPLETE {len(results)}/{len(results)}',flush=True)

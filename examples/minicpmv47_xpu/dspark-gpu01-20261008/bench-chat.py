import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

root = Path(__file__).resolve().parent
profile = sys.argv[1]
base = 'http://127.0.0.1:31065'
plan = json.loads((root/'PLAN.json').read_text())
results = json.loads((root/f'{profile}-correctness.json').read_text())
assert len(results) == 6
if not all(row['passed'] for row in results):
    assert profile == 'dspark_native'
    assert all(row['http_status']==200 and row['expected_ok'] for row in results)
    assert all(row['passed'] for row in results if row['case']!='context_8192_decode64')
    eligibility={'evidence_class':'development_conditional_latency',
                 'benchmark_context_tokens':2048,
                 'benchmark_context_matches_baseline':True,
                 'open_limitation':'8192 context token divergence at index 23; candidate comma and full stop tied in three diagnostics',
                 'formal_acceptance':False}
    (root/f'{profile}-performance-eligibility.json').write_text(json.dumps(eligibility,indent=2)+'\n')
case = next(
    case for case in json.loads((root/'correctness-cases.json').read_text())
    if case.get('prompt_tokens') == 2048)
prompt = case['text']
body = {'model':'MiniCPM-V-4.7-1008',
        'messages':[{'role':'user','content':prompt}],
        'temperature':0, 'seed':7, 'min_tokens':256, 'max_tokens':256,
        'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True},
        'chat_template_kwargs':{'enable_thinking':False}}

def metrics():
    with urllib.request.urlopen(base+'/metrics',timeout=10) as response:
        return response.read().decode()

samples = []
for index in range(plan['warmup_requests'] + plan['measured_requests']):
    before = metrics()
    request = urllib.request.Request(base+'/v1/chat/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start = time.perf_counter()
    first = done = None
    chunks = []
    usage = None
    request_metrics = None
    with urllib.request.urlopen(request,timeout=300) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            payload=raw[len(b'data: '):].strip()
            now=time.perf_counter()
            if payload == b'[DONE]':
                done=now
                break
            data=json.loads(payload)
            usage=data.get('usage') or usage
            request_metrics=data.get('metrics') or request_metrics
            for choice in data.get('choices',[]):
                content=choice.get('delta',{}).get('content')
                if content:
                    if first is None:
                        first=now
                    chunks.append(content)
    assert first is not None and done is not None and usage is not None
    assert usage['prompt_tokens']==2048 and usage['completion_tokens']==256,usage
    result={'phase':'warmup' if index<plan['warmup_requests'] else 'measured',
            'index':index,'usage':usage,'content_chunks':len(chunks),
            'ttft_ms':(first-start)*1000,'tpot_ms':(done-first)*1000/255,
            'total_ms':(done-start)*1000,'generated_text':''.join(chunks),
            'request_metrics':request_metrics}
    samples.append(result)
    (root/f'{profile}-samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    (root/f'{profile}-metrics-{index}-before.txt').write_text(before)
    (root/f'{profile}-metrics-{index}-after.txt').write_text(metrics())
    print(profile,result['phase'],index,'TTFT',round(result['ttft_ms'],3),
          'TPOT',round(result['tpot_ms'],3),'total',round(result['total_ms'],3),flush=True)
measured=[row for row in samples if row['phase']=='measured']
summary={'profile':profile,'measured_samples':len(measured),
         'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
         'medians':{key:statistics.median(row[key] for row in measured)
                    for key in ['ttft_ms','tpot_ms','total_ms']}}
(root/f'{profile}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE',json.dumps(summary),flush=True)

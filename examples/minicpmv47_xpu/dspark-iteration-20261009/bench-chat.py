import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

root=Path(__file__).resolve().parent
label=sys.argv[1]
plan=json.loads((root/'PLAN.json').read_text())
receipt=json.loads((root/f'{label}-launch.json').read_text())
assert receipt.get('profile',label) in plan['initial_profiles']
base='http://127.0.0.1:31065'
model='MiniCPM-V-4.7-1008'

def post(path,body,timeout=300):
    request=urllib.request.Request(base+path,data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(request,timeout=timeout)

for _ in range(600):
    server_log=root.parent/'dspark-gpu01-20261008'/Path(receipt['log']).name
    if server_log.exists() and 'RuntimeError: Worker failed with error' in server_log.read_text():
        raise RuntimeError(f'Server startup failed; see {server_log}')
    try:
        with urllib.request.urlopen(base+'/health',timeout=2) as response:
            if response.status==200:
                break
    except (urllib.error.URLError,TimeoutError):
        time.sleep(2)
else:
    raise RuntimeError('Service readiness timeout')
print(label,'ready',flush=True)
sanity=[]
for prompt,expected in [('计算 17 × 24，只回答数字。','408'),
                         ('中国的首都是哪里？只回答城市名称。','北京'),
                         ('Translate 苹果 into English. Answer with one word.','apple'),
                         ('What is the capital of France? Answer with one word.','paris')]:
    with post('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':prompt}],
              'temperature':0,'seed':7,'max_tokens':64,
              'chat_template_kwargs':{'enable_thinking':False}}) as response:
        data=json.load(response)
    text=data['choices'][0]['message']['content'] or ''
    row={'prompt':prompt,'passed':expected.lower() in text.lower(),'response':data}
    sanity.append(row)
    print(label,'warmup sanity',row['passed'],repr(text[:80]),flush=True)
(root/f'{label}-sanity.json').write_text(json.dumps(sanity,ensure_ascii=False,indent=2)+'\n')
assert all(row['passed'] for row in sanity),'Gross functional issue before performance'
cases=json.loads((root.parent/'dspark-gpu01-20261008/correctness-cases.json').read_text())
prompt=next(row['text'] for row in cases if row.get('prompt_tokens')==plan['prompt_tokens'])
assert hashlib.sha256(prompt.encode()).hexdigest()==plan['prompt_sha256']
body={'model':model,'messages':[{'role':'user','content':prompt}],
      'temperature':0,'seed':7,'min_tokens':256,'max_tokens':256,'ignore_eos':True,
      'stream':True,'stream_options':{'include_usage':True},
      'chat_template_kwargs':{'enable_thinking':False}}

def metrics():
    with urllib.request.urlopen(base+'/metrics',timeout=10) as response:
        return response.read().decode()

samples=[]
for index in range(plan['full_prompt_warmup_requests']+plan['measured_requests']):
    before=metrics()
    start=time.perf_counter()
    first=done=None
    usage=request_metrics=None
    chunks=[]
    with post('/v1/chat/completions',body) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            payload=raw[len(b'data: '):].strip()
            now=time.perf_counter()
            if payload==b'[DONE]':
                done=now
                break
            data=json.loads(payload)
            usage=data.get('usage') or usage
            request_metrics=data.get('metrics') or request_metrics
            for choice in data.get('choices',[]):
                text=choice.get('delta',{}).get('content')
                if text:
                    if first is None:
                        first=now
                    chunks.append(text)
    assert first is not None and done is not None and usage is not None
    assert usage['prompt_tokens']==2048 and usage['completion_tokens']==256,usage
    row={'phase':'warmup' if index<plan['full_prompt_warmup_requests'] else 'measured',
         'index':index,'usage':usage,'content_chunks':len(chunks),
         'ttft_ms':(first-start)*1000,'tpot_ms':(done-first)*1000/255,
         'total_ms':(done-start)*1000,'generated_text':''.join(chunks),
         'request_metrics':request_metrics}
    samples.append(row)
    (root/f'{label}-samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    (root/f'{label}-metrics-{index}-before.txt').write_text(before)
    (root/f'{label}-metrics-{index}-after.txt').write_text(metrics())
    print(label,row['phase'],index,'TTFT',round(row['ttft_ms'],3),
          'TPOT',round(row['tpot_ms'],3),'total',round(row['total_ms'],3),flush=True)
measured=[row for row in samples if row['phase']=='measured']
summary={'profile':label,'measured_samples':len(measured),'prompt_sha256':plan['prompt_sha256'],
         'medians':{key:statistics.median(row[key] for row in measured)
                    for key in ['ttft_ms','tpot_ms','total_ms']}}
acceptance=[row['request_metrics']['speculative_decoding'] for row in measured
            if row['request_metrics'] and row['request_metrics'].get('speculative_decoding')]
if acceptance:
    steps=sum(row['num_spec_steps'] for row in acceptance)
    accepted=sum(row['num_accepted_draft_tokens'] for row in acceptance)
    proposed=sum(row['num_draft_tokens'] for row in acceptance)
    summary['speculative']={'accepted':accepted,'proposed':proposed,'steps':steps,
                            'acceptance_rate':accepted/proposed,'mean_acceptance_length':1+accepted/steps}
(root/f'{label}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE',json.dumps(summary),flush=True)

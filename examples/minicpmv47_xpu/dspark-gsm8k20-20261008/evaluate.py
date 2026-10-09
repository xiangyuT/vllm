import json
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

from helpers import get_answer_value

root=Path(__file__).resolve().parent
profile=sys.argv[1]
plan_file=sys.argv[2] if len(sys.argv)>2 else 'PLAN.json'
plan=json.loads((root/plan_file).read_text())
assert profile in plan['profiles']
questions=json.loads((root/'questions.json').read_text())
base='http://127.0.0.1:31065'
output=root/f'{profile}-results.json'
assert not output.exists()

def post(body):
    request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:
        return response.status,json.load(response)

for _ in range(300):
    try:
        with urllib.request.urlopen(base+'/health',timeout=2) as response:
            if response.status==200:
                break
    except (urllib.error.URLError,TimeoutError):
        time.sleep(2)
else:
    raise RuntimeError('Server readiness timeout')

common={'model':'MiniCPM-V-4.7-1008','temperature':plan['temperature'],
        'seed':plan['seed'],'max_tokens':plan['max_tokens'],
        'chat_template_kwargs':plan['chat_template_kwargs'],'return_token_ids':True}
_,warmup=post({**common,'messages':[{'role':'user','content':'Calculate 1 + 1. Answer with one number.'}],
              'max_tokens':32})
(root/f'{profile}-warmup.json').write_text(json.dumps(warmup,ensure_ascii=False,indent=2)+'\n')
results=[]
reuse={}
if len(sys.argv)>3:
    previous=json.loads((root/sys.argv[3]).read_text())
    reuse={row['test_index']:row for row in previous
           if row['finish_reason']=='stop' and row['http_status']==200}
evaluation_start=time.perf_counter()
for question in questions:
    if question['test_index'] in reuse:
        row=dict(reuse[question['test_index']])
        assert row['prompt_sha256']==question['prompt_sha256'] and row['label']==question['label']
        row.update(reused=True,source_results=sys.argv[3],generation_max_tokens=256)
        results.append(row)
        output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(profile,f'{len(results)}/20','REUSED EOS',
              'CORRECT' if row['correct'] else 'WRONG','label',row['label'],flush=True)
        continue
    start=time.perf_counter()
    status,response=post({**common,'messages':[{'role':'user','content':question['prompt']}]})
    latency=time.perf_counter()-start
    choice=response['choices'][0]
    text=choice['message']['content'] or ''
    prediction=get_answer_value(text)
    row={'test_index':question['test_index'],'prompt_sha256':question['prompt_sha256'],
         'label':question['label'],'prediction':prediction,
         'correct':prediction==question['label'],'http_status':status,
         'finish_reason':choice['finish_reason'],'latency_seconds':latency,'response':response,
         'reused':False,'generation_max_tokens':plan['max_tokens']}
    results.append(row)
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(profile,f'{len(results)}/20','CORRECT' if row['correct'] else 'WRONG',
          'label',row['label'],'pred',prediction,'tokens',response['usage']['completion_tokens'],
          'seconds',round(latency,2),'finish',row['finish_reason'],flush=True)
summary={'profile':profile,'num_questions':len(results),
         'correct':sum(row['correct'] for row in results),
         'accuracy':sum(row['correct'] for row in results)/len(results),
         'invalid_answers':sum(row['prediction']==-9999999 for row in results),
         'length_truncations':sum(row['finish_reason']=='length' for row in results),
         'total_output_tokens':sum(row['response']['usage']['completion_tokens'] for row in results),
         'evaluation_wall_seconds':time.perf_counter()-evaluation_start,
         'median_question_seconds':statistics.median(row['latency_seconds'] for row in results),
         'max_concurrency':1,'num_shots':5,'temperature':0,'seed':7,'max_tokens':plan['max_tokens'],
         'reused_natural_eos_results':sum(row.get('reused',False) for row in results),
         'executed_questions_this_run':sum(not row.get('reused',False) for row in results),
         'timing_note':'wall time includes newly executed questions only; reused per-question timings retain their original provenance'}
(root/f'{profile}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('COMPLETE',json.dumps(summary),flush=True)

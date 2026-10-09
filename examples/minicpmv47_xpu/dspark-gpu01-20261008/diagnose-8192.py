import json
from pathlib import Path
import sys
import urllib.request

root=Path(__file__).resolve().parent
label=sys.argv[1]
context=int(sys.argv[2]) if len(sys.argv)>2 else 8192
max_tokens=32 if context==8192 else 128
case=next(row for row in json.loads((root/'correctness-cases.json').read_text())
          if row['case']==f'context_{context}_decode64')
reference=next(row for row in json.loads((root/'baseline_eager-correctness.json').read_text())
               if row['case']==case['case'])
results=[]
for trial in range(3):
    body={'model':'MiniCPM-V-4.7-1008',
          'messages':[{'role':'user','content':case['text']}],
          'temperature':0,'seed':7,'max_tokens':max_tokens,'min_tokens':max_tokens,'ignore_eos':True,
          'return_token_ids':True,'return_tokens_as_token_ids':True,
          'logprobs':True,'top_logprobs':5,
          'chat_template_kwargs':{'enable_thinking':False}}
    request=urllib.request.Request('http://127.0.0.1:31065/v1/chat/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:
        data=json.load(response)
    choice=data['choices'][0]
    ids=choice['token_ids']
    difference=next((i for i,(a,b) in enumerate(zip(ids,reference['token_ids'])) if a!=b),None)
    result={'trial':trial,'first_difference':difference,
            'position23':choice['logprobs']['content'][23],'response':data}
    results.append(result)
    (root/f'{label}-{context}-diagnostics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(label,trial,'first difference',difference,
          'position23',json.dumps(result['position23'],ensure_ascii=False),flush=True)
print('Repeated outputs identical',all(row['response']['choices'][0]['token_ids']==results[0]['response']['choices'][0]['token_ids'] for row in results),flush=True)

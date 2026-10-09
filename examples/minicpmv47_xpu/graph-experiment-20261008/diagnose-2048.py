import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

root = Path(__file__).resolve().parent
label = sys.argv[1]
assert label in ['FULL_DECODE_ONLY', 'NONE_RECHECK']
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen('http://127.0.0.1:31063/health', timeout=2) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(3)
else:
    raise RuntimeError('service readiness timeout')
case = next(row for row in json.loads((root / 'correctness-cases.json').read_text())
            if row['case'] == 'context_2048_decode64')
reference = next(row for row in json.loads((root / 'NONE-correctness.json').read_text())
                 if row['case'] == case['case'])
results = []
for trial in range(3):
    body = {'model': 'MiniCPM-V-4.7-old',
            'messages': [{'role': 'user', 'content': case['text']}],
            'temperature': 0, 'seed': 7, 'max_tokens': 64, 'min_tokens': 64,
            'ignore_eos': True, 'return_token_ids': True,
            'return_tokens_as_token_ids': True, 'logprobs': True, 'top_logprobs': 5,
            'chat_template_kwargs': {'enable_thinking': False}}
    request = urllib.request.Request('http://127.0.0.1:31063/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.load(response)
    choice = data['choices'][0]
    ids = choice['token_ids']
    difference = next((i for i,(x,y) in enumerate(zip(ids,reference['token_ids'])) if x!=y), None)
    at10 = choice['logprobs']['content'][10]
    row = {'trial': trial, 'matches_original_NONE': ids == reference['token_ids'],
           'first_difference': difference, 'position_10': at10, 'response': data}
    results.append(row)
    (root / f'{label}-2048-diagnostics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(label, trial, 'matches NONE', row['matches_original_NONE'],
          'position10', json.dumps(at10, ensure_ascii=False), flush=True)
print('Repeated outputs identical:', all(row['response']['choices'][0]['token_ids'] == results[0]['response']['choices'][0]['token_ids'] for row in results), flush=True)

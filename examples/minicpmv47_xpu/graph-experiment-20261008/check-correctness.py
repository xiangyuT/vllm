import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

root = Path(__file__).resolve().parent
profile = sys.argv[1]
assert profile in ['NONE', 'PIECEWISE', 'FULL_DECODE_ONLY']
base = 'http://127.0.0.1:31063'
model = 'MiniCPM-V-4.7-old'

def post(endpoint, body, timeout=180):
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
    raise RuntimeError('service not ready within 15 minutes')
print(profile, 'service ready', flush=True)

def tokenize(text):
    _, data = post('/tokenize', {'model': model,
        'messages': [{'role': 'user', 'content': text}], 'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False}})
    return data['count']

cases_path = root / 'correctness-cases.json'
if not cases_path.exists():
    assert profile == 'NONE'
    cases = [
        {'case': 'arithmetic', 'text': '计算 17 × 24，只回答数字。', 'expected': '408', 'force64': False},
        {'case': 'capital', 'text': '中国的首都是哪里？只回答城市名称。', 'expected': '北京', 'force64': False},
        {'case': 'translation', 'text': 'Translate 苹果 into English. Answer with one word.', 'expected': 'apple', 'force64': False}]
    full = (root / 'prompt.txt').read_text()
    for target in [512, 2048, 8192, 32768]:
        if target == 32768:
            text = full
        else:
            low, high = 1, len(full)
            while low < high:
                middle = (low + high) // 2
                if tokenize(full[:middle]) < target:
                    low = middle + 1
                else:
                    high = middle
            text = next((full[:length] for length in range(max(1,low-32),low+64)
                         if tokenize(full[:length]) == target), None)
            assert text is not None
        assert tokenize(text) == target
        cases.append({'case': f'context_{target}_decode64', 'text': text,
                      'prompt_tokens': target, 'expected': None, 'force64': True})
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
else:
    cases = json.loads(cases_path.read_text())
reference = None if profile == 'NONE' else json.loads((root / 'NONE-correctness.json').read_text())
results = []
for index, case in enumerate(cases):
    body = {'model': model, 'messages': [{'role': 'user', 'content': case['text']}],
            'temperature': 0, 'seed': 7, 'max_tokens': 64, 'return_token_ids': True,
            'chat_template_kwargs': {'enable_thinking': False}}
    if case['force64']:
        body.update(min_tokens=64, ignore_eos=True)
    status, response = post('/v1/chat/completions', body)
    choice = response['choices'][0]
    text = choice['message']['content'] or ''
    token_ids = choice.get('token_ids')
    assert token_ids is not None and len(token_ids) == response['usage']['completion_tokens']
    expected_ok = case['expected'] is None or case['expected'].lower() in text.lower()
    if case['force64']:
        assert response['usage']['prompt_tokens'] == case['prompt_tokens']
        assert response['usage']['completion_tokens'] == 64
    token_match = reference is None or token_ids == reference[index]['token_ids']
    result = {'case': case['case'], 'http_status': status, 'response': response,
              'token_ids': token_ids, 'expected_ok': expected_ok,
              'matches_NONE_token_ids': None if reference is None else token_match,
              'passed': status == 200 and expected_ok and token_match}
    if reference is not None and not token_match:
        ref_ids = reference[index]['token_ids']
        result['first_token_difference'] = next((i for i,(a,b) in enumerate(zip(token_ids,ref_ids)) if a!=b), min(len(token_ids),len(ref_ids)))
    results.append(result)
    (root / f'{profile}-correctness.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(profile, case['case'], 'PASS' if result['passed'] else 'FAIL',
          'tokens', len(token_ids), 'answer', repr(text[:80]), flush=True)
assert all(row['passed'] for row in results), 'correctness mismatch; latency collection withheld'
print(profile, f'COMPLETE: {len(results)}/{len(results)} correctness checks passed', flush=True)

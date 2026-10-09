import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

root = Path(__file__).resolve().parent
profile = sys.argv[1]
base = 'http://127.0.0.1:31063'
model = 'MiniCPM-V-4.7-old'
plan = json.loads((root / 'PLAN.json').read_text())

def post(path, body, timeout=180):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(base + '/health', timeout=2) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(3)
else:
    raise RuntimeError('service readiness timeout')
print(profile, 'service ready', flush=True)

sanity = []
for prompt, expected in [('计算 17 × 24，只回答数字。', '408'),
                         ('中国的首都是哪里？只回答城市名称。', '北京')]:
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 64, 'seed': 7,
            'chat_template_kwargs': {'enable_thinking': False}}
    with post('/v1/chat/completions', body) as response:
        data = json.load(response)
    answer = data['choices'][0]['message']['content'] or ''
    sanity.append({'prompt': prompt, 'response': data, 'passed': expected in answer})
    print(profile, 'sanity', repr(answer), flush=True)
(root / f'{profile}-sanity.json').write_text(json.dumps(sanity, ensure_ascii=False, indent=2) + '\n')
assert all(item['passed'] for item in sanity), 'sanity failed; performance collection cancelled'

def token_count(text):
    body = {'model': model, 'messages': [{'role': 'user', 'content': text}],
            'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False}}
    with post('/tokenize', body) as response:
        return json.load(response)['count']

prompt_file = root / 'prompt.txt'
if prompt_file.exists():
    prompt = prompt_file.read_text()
    assert token_count(prompt) == plan['prompt_tokens']
else:
    head = '以下是编号资料，仅用于测试长上下文推理延迟。末尾请简短回答。\n'
    full = head + ''.join(f'编号{i:05d}：这是用于长上下文延迟测试的重复资料。\n' for i in range(12000))
    lo, hi = len(head), len(full)
    while lo < hi:
        mid = (lo + hi) // 2
        count = token_count(full[:mid])
        if count < plan['prompt_tokens']:
            lo = mid + 1
        else:
            hi = mid
    found = None
    for length in range(max(len(head), lo - 32), min(len(full), lo + 64)):
        if token_count(full[:length]) == plan['prompt_tokens']:
            found = full[:length]
            break
    assert found is not None, 'could not form exact 32768-token chat input'
    prompt = found
    prompt_file.write_text(prompt)
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    (root / 'prompt-manifest.json').write_text(json.dumps({
        'sha256_utf8': sha, 'characters': len(prompt), 'prompt_tokens': token_count(prompt),
        'historical_sha256': plan['historical_prompt_sha256'],
        'matches_historical_prompt': sha == plan['historical_prompt_sha256'],
        'construction': 'numbered repeated Chinese lines, character-truncated using server /tokenize'}, indent=2) + '\n')

body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'seed': 7, 'min_tokens': 512, 'max_tokens': 512,
        'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False}}
samples = []
for index in range(plan['warmup_requests'] + plan['measured_requests']):
    start = time.perf_counter()
    first = done = None
    chunks = []
    usage = None
    with post('/v1/chat/completions', body, timeout=300) as response:
        status = response.status
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            payload = raw[len(b'data: '):].strip()
            now = time.perf_counter()
            if payload == b'[DONE]':
                done = now
                break
            data = json.loads(payload)
            usage = data.get('usage') or usage
            for choice in data.get('choices', []):
                content = choice.get('delta', {}).get('content')
                if content:
                    if first is None:
                        first = now
                    chunks.append(content)
    assert first is not None and done is not None and usage is not None
    assert usage['prompt_tokens'] == 32768 and usage['completion_tokens'] == 512, usage
    result = {'phase': 'warmup' if index < 2 else 'measured', 'index': index,
              'http_status': status, 'usage': usage, 'content_chunks': len(chunks),
              'ttft_ms': (first - start) * 1000,
              'tpot_ms': (done - first) * 1000 / 511,
              'total_ms': (done - start) * 1000,
              'generated_text': ''.join(chunks)}
    samples.append(result)
    (root / f'{profile}-samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
    print(profile, result['phase'], index, 'TTFT', round(result['ttft_ms'], 3),
          'TPOT', round(result['tpot_ms'], 3), 'total', round(result['total_ms'], 3), flush=True)
measured = [row for row in samples if row['phase'] == 'measured']
summary = {'profile': profile, 'measured_samples': len(measured),
           'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
           'medians': {key: statistics.median(row[key] for row in measured)
                       for key in ['ttft_ms', 'tpot_ms', 'total_ms']}}
(root / f'{profile}-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('COMPLETE', json.dumps(summary), flush=True)

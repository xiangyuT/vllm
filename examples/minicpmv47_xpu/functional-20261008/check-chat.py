import json
from pathlib import Path
import time
import urllib.error
import urllib.request

root = Path(__file__).resolve().parent
base = 'http://127.0.0.1:31063'
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(base + '/health', timeout=2) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(3)
else:
    raise RuntimeError('server did not become healthy within 10 minutes')
print('Server health HTTP 200', flush=True)
cases = [('arithmetic', '计算 17 × 24，只回答数字。', '408'),
         ('chinese', '中国的首都是哪里？只回答城市名称。', '北京'),
         ('english', 'Translate 苹果 into English. Answer with one word.', 'apple')]
results = []
for name, prompt, expected in cases:
    body = {'model': 'MiniCPM-V-4.7-old', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 64, 'seed': 7,
            'chat_template_kwargs': {'enable_thinking': False}}
    request = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.load(response)
        result = {'case': name, 'request': body, 'http_status': response.status,
                  'response': data, 'elapsed_seconds': time.monotonic() - start}
    text = data['choices'][0]['message']['content'] or ''
    result['expected_substring'] = expected
    result['expected_found'] = expected.lower() in text.lower()
    results.append(result)
    (root / 'chat-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(name, repr(text), result['expected_found'], flush=True)
assert all(r['expected_found'] for r in results), 'short-answer functional check failed'
print('All 3 short-answer functional checks passed', flush=True)

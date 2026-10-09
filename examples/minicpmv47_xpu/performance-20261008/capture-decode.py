import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent
base = 'http://127.0.0.1:31063'
def post(endpoint, body):
    request = urllib.request.Request(base + endpoint, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=180)

with post('/start_profile', {'profile_prefix': 'bf16_decode', 'delay_iterations': 8, 'max_iterations': 4}) as response:
    assert response.status == 200
with post('/v1/chat/completions', {
    'model': 'MiniCPM-V-4.7-old',
    'messages': [{'role': 'user', 'content': (root / 'prompt.txt').read_text()}],
    'temperature': 0, 'seed': 7, 'max_tokens': 16, 'min_tokens': 16, 'ignore_eos': True,
    'chat_template_kwargs': {'enable_thinking': False}}) as response:
    data = json.load(response)
    assert data['usage']['prompt_tokens'] == 32768
    assert data['usage']['completion_tokens'] == 16
(root / 'profile-request.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
with post('/stop_profile', {}) as response:
    assert response.status == 200
print('Decode capture complete; excluded from performance samples', flush=True)

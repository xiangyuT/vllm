import hashlib
import json
from pathlib import Path

from helpers import build_prompts, source

root=Path(__file__).resolve().parent
plan=json.loads((root/'PLAN.json').read_text())
train=[json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
test=[json.loads(line) for line in (root/'test.jsonl').read_text().splitlines()]
assert len(train)==7473 and len(test)==1319
assert not ({row['question'] for row in train[:5]} & {row['question'] for row in test[:20]})
prompts,labels=build_prompts(train,test,plan['num_questions'],plan['num_shots'])
rows=[{'test_index':i,'question':test[i]['question'],
       'reference_answer':test[i]['answer'],'label':label,'prompt':prompt,
       'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
      for i,(prompt,label) in enumerate(zip(prompts,labels))]
assert len(rows)==20
(root/'questions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
manifest={'dataset_urls':{name:plan['dataset_source']+name for name in ['train.jsonl','test.jsonl']},
          'dataset_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['train.jsonl','test.jsonl']},
          'selected_test_indices':list(range(20)),
          'questions_sha256':hashlib.sha256((root/'questions.json').read_bytes()).hexdigest(),
          'canonical_helper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'scoring':'unmodified upstream get_answer_value: remove commas, parse last integer; exact equality to reference',
          'num_shots':5,'num_questions':20,'gpu_execution':False}
(root/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared identical 20 test questions with upstream 5-shot prompts and scoring; labels',labels)

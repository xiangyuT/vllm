import ast
import json
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parent
repo = root.parent / 'vllm-functional'
draft = json.loads((root / 'checkpoint-configs.json').read_text())['config.json']
target = json.loads(Path('/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-1008/config.json').read_text())['text_config']
headers = json.loads((root / 'safetensors-headers.json').read_text())
namespace = {}
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
for file, name in [
    ('vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py', 'get_eagle3_aux_layers_from_config'),
    ('vllm/model_executor/models/qwen3_dflash.py', '_get_dflash_fc_input_size')]:
    tree = ast.parse((repo / file).read_text())
    nodes.append(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name))
module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
exec(compile(module, '<extracted-pure-config-functions>', 'exec'), namespace)
hf = SimpleNamespace(**draft)
spec = SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf))
aux_layers = namespace['get_eagle3_aux_layers_from_config'](spec)
fc_width = namespace['_get_dflash_fc_input_size'](SimpleNamespace(speculative_config=spec))
assert aux_layers == (4,12,20,28,36)
assert fc_width == headers['fc.weight']['shape'][1] == 10240
assert draft['hidden_size'] == target['hidden_size'] == 2048
assert draft['vocab_size'] == draft['draft_vocab_size'] == target['vocab_size'] == 248144
assert draft['num_target_layers'] == target['num_hidden_layers'] == 40
assert headers['confidence_head.proj.weight']['shape'] == [1, draft['hidden_size']+draft['markov_rank']]
assert headers['markov_head.markov_w1.weight']['shape'] == [248144,256]
assert headers['markov_head.markov_w2.weight']['shape'] == [248144,256]
assert all(target['layer_types'][i]=='full_attention' for i in draft['target_layer_ids'])
assert not any('embed_tokens' in name or 'lm_head' in name for name in headers)
report = {'gpu_execution':False,'torch_imported':False,
          'evidence_class':'checkpoint_structure_and_extracted_pure_config_helpers',
          'aux_hidden_state_capture_indices':aux_layers,'fc_input_width':fc_width,
          'target_layer_ids':draft['target_layer_ids'],
          'selected_target_layer_types':[target['layer_types'][i] for i in draft['target_layer_ids']],
          'target_and_draft_vocab':248144,'target_and_draft_hidden_size':2048,
          'checkpoint_tensor_parameters':sum((spec['data_offsets'][1]-spec['data_offsets'][0])//2 for spec in headers.values()),
          'draft_rotary_dim':draft['rope_parameters']['rope_dim'],
          'target_rotary_dim':int(target['head_dim']*target['rope_parameters']['partial_rotary_factor']),
          'confidence_input_width':2304,'block_size':15,'default_target_verification_query_length':16,
          'embeddings_and_lm_head_in_checkpoint':False,
          'markov_projection_weight_bytes':248144*256*2,
          'model_loaded':False}
(root / 'static-contract-check.json').write_text(json.dumps(report,indent=2) + '\n')
print(json.dumps(report,indent=2))

from collections import Counter
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parent
results = []
keywords = ['xpu_moe_shared_fused', 'moe_shared_fused_decode_interface',
            'router_resadd_norm_gemv_topk_softmax', 'router_gemv_topk_softmax',
            'qkv_split_norm_rope', 'fp8_gemm_w8a16_pair', 'gated_rmsnorm_fp8_gemm',
            'resadd_rmsnorm_fp8_gemm', 'xpu_all_reduce_', 'gdn_attention_core_xpu']
for path in sorted((root / 'traces').glob('*.json.gz')):
    with gzip.open(path, 'rt') as handle:
        events = json.load(handle)['traceEvents']
    operators = Counter(event.get('name', '') for event in events
                        if event.get('cat') == 'cpu_op')
    kernels = Counter(event.get('name', '') for event in events
                      if event.get('cat') in ['kernel', 'xpu_kernel'])
    matched = {name: count for name, count in operators.items()
               if any(key in name for key in keywords)}
    decode_events = []
    for event in events:
        dims = event.get('args', {}).get('Input Dims')
        if event.get('cat') == 'cpu_op' and dims and dims[0] and dims[0][0] == 1:
            decode_events.append(event)
    decode_counts = Counter(event['name'] for event in decode_events)
    decode_matched = {name: count for name, count in decode_counts.items()
                      if any(key in name for key in keywords)}
    kernel_matches = {name: count for name, count in kernels.items()
                      if any(key in name.lower() for key in ['gemv', 'moeshared', 'moe_shared', 'qkv', 'router'])}
    sample_shapes = {name: [event.get('args', {}).get('Input Dims')
                           for event in events if event.get('name') == name][:2]
                     for name in matched}
    results.append({'trace': path.name, 'event_count': len(events),
                    'categories': dict(Counter(event.get('cat', '') for event in events)),
                    'operators': matched, 'decode_operators_m1': decode_matched,
                    'sample_input_shapes': sample_shapes,
                    'native_kernels': kernel_matches})
assert results, 'no profiler trace found'
(root / 'trace-summary.json').write_text(json.dumps(results, indent=2) + '\n')
for result in results:
    print(result['trace'], 'M=1 operators', json.dumps(result['decode_operators_m1']),
          'matching kernel kinds', len(result['native_kernels']))

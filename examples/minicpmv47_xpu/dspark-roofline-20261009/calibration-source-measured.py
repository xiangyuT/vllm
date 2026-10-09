"""Short, three-sample XPU reference measurements; no model loading."""

import datetime
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

OUT = Path('/audit')
EXPECTED = ['868023e2-0000-0000-1800-000000000000',
            '868023e2-0000-0000-3600-000000000000']


def identity():
    assert torch.xpu.device_count() == 2
    result = []
    for d in range(2):
        p = torch.xpu.get_device_properties(d)
        assert str(p.uuid) == EXPECTED[d], (d, str(p.uuid))
        result.append({'logical': d, 'uuid': str(p.uuid), 'name': p.name})
    return result


def measure(fn, repeats, device, graph=False):
    torch.xpu.set_device(device)
    for _ in range(4):
        fn()
    torch.xpu.synchronize(device)
    captured = None
    if graph:
        captured = torch.xpu.XPUGraph()
        stream = torch.xpu.Stream(device=device)
        stream.wait_stream(torch.xpu.current_stream(device))
        with torch.xpu.graph(captured, stream=stream):
            for _ in range(repeats):
                fn()
        captured.replay()
        torch.xpu.synchronize(device)
    rows = []
    for _ in range(3):
        torch.xpu.synchronize(device)
        start = time.perf_counter()
        if captured is None:
            for _ in range(repeats):
                fn()
        else:
            captured.replay()
        torch.xpu.synchronize(device)
        rows.append((time.perf_counter() - start) * 1e6 / repeats)
    return {'samples_us': rows, 'median_us': statistics.median(rows),
            'repeats_per_sample': repeats, 'graph': graph,
            'time_boundary': 'wall time; synchronized block / repeats'}


def compute():
    identities = identity()
    rows = []
    for device in range(2):
        torch.xpu.set_device(device)
        torch.manual_seed(7)
        torch.xpu.manual_seed_all(7)
        n = 128 * 1024 * 1024
        a = torch.full((n,), 1.25, device=f'xpu:{device}')
        b = torch.empty_like(a)
        r = measure(lambda: b.copy_(a), 20, device, graph=True)
        assert b[0].item() == 1.25 and b[-1].item() == 1.25
        r.update({'kind': 'stream_copy', 'logical_device': device,
                  'read_plus_write_bytes': 2 * n * 4,
                  'reference_bandwidth_bytes_per_second': 2*n*4/(r['median_us']*1e-6)})
        rows.append(r)
        del a, b
        torch.xpu.empty_cache()
        for shape in [(4096,4096,4096), (1,256,248192)]:
            m,k,n = shape
            a = torch.randn((m,k), device=f'xpu:{device}', dtype=torch.bfloat16)
            # Same column-major W view as the retained Markov aten::mm trace.
            w = torch.randn((n,k), device=f'xpu:{device}', dtype=torch.bfloat16).t()
            c = torch.empty((m,n), device=f'xpu:{device}', dtype=torch.bfloat16)
            r = measure(lambda: torch.mm(a,w,out=c), 20, device, graph=True)
            assert torch.isfinite(c).all().item()
            r.update({'kind': 'bf16_gemm_reference', 'logical_device': device,
                      'mnk': [m,n,k], 'weight_stride': list(w.stride()),
                      'flops_2mnk': 2*m*n*k,
                      'reference_flop_per_second': 2*m*n*k/(r['median_us']*1e-6)})
            rows.append(r)
            del a,w,c
            torch.xpu.empty_cache()
        print('compute rank', device, 'complete', flush=True)
    result = {'evidence_class': 'development_hardware_reference',
              'identities': identities, 'seed': 7, 'torch': torch.__version__,
              'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'rows': rows,
              'limits': ['attainable references, not measured instruction/DRAM counters',
                         'three measured blocks after warmup; no model latency samples',
                         'reference GEMM ceiling is shape dependent; ratio above one is not clipped']}
    tag = os.environ.get('CALIBRATE_TAG', '')
    (OUT/f'compute-reference{tag}.json').write_text(json.dumps(result,indent=2)+'\n')


def collective(rank, port):
    identity()
    torch.xpu.set_device(rank)
    dist.init_process_group('xccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank,world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    cpu = dist.new_group([0,1],backend='gloo',
                         timeout=datetime.timedelta(seconds=90))
    rows = []
    # Match XpuCommunicator's clone + blocking all_reduce; input never doubles.
    for m in [1,3,4,7,8,15,16]:
        a = torch.full((m,2048),float(rank+1),device=f'xpu:{rank}',dtype=torch.bfloat16)
        output = torch.empty_like(a)
        def fn():
            output.copy_(a)
            dist.all_reduce(output)
        for graph in [False,True]:
            dist.barrier(group=cpu)
            r = measure(fn,20,rank,graph=graph)
            assert torch.all(output==3).item()
            r.update({'kind':'all_reduce','logical_device':rank,'rows':m,
                      'payload_bytes':a.numel()*a.element_size()})
            rows.append(r)
        print('allreduce',rank,m,'complete',flush=True)
    for m in [1,3,4,7,8,15,16]:
        a = torch.full((m,124096),float(rank+1),device=f'xpu:{rank}',dtype=torch.bfloat16)
        output = torch.empty((2*m,124096),device=f'xpu:{rank}',dtype=torch.bfloat16)
        def fn():
            dist.all_gather_into_tensor(output,a)
        for graph in [False,True]:
            dist.barrier(group=cpu)
            r = measure(fn,10,rank,graph=graph)
            assert torch.all(output[:m]==1).item() and torch.all(output[m:]==2).item()
            r.update({'kind':'vocab_all_gather','logical_device':rank,'rows':m,
                      'payload_bytes':a.numel()*a.element_size()})
            rows.append(r)
        print('allgather',rank,m,'complete',flush=True)
    (OUT/f'collective-reference-rank{rank}.json').write_text(json.dumps({
        'evidence_class':'development_balanced_collective_reference',
        'identities':identity(),'rows':rows,
        'limits':['balanced synthetic arrivals; no model peer-wait attribution',
                  'graph batches aggregate 10/20 collectives; not single-call model graph latency',
                  'all-reduce copy uses persistent output; allocator cost excluded']},indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    if os.environ.get('CALIBRATE_MODE')=='collective':
        mp.spawn(collective,args=(31209,),nprocs=2,join=True)
    else:
        compute()

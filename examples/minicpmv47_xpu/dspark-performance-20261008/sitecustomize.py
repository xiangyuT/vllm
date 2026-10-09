import importlib.util
import os
import sys

path='/workspace/preparation/performance-20261008/sitecustomize.py'
spec=importlib.util.spec_from_file_location('minicpm_bf16_startup',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if os.environ.get('MINICPM_DSPARK_ATTENTION')=='1':
    sys.path.insert(0,'/workspace/preparation/dspark-gpu01-20261008')
    import dspark_attention  # noqa: F401
if os.environ.get('MINICPM_CG_AUDIT')=='1':
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    original_fullgraph=CudaGraphManager.run_fullgraph

    def audit_fullgraph(self,desc):
        result=original_fullgraph(self,desc)
        if not getattr(self,'_minicpm_logged_replay',False):
            self._minicpm_logged_replay=True
            print('MINICPM FULL GRAPH REPLAY',type(self).__name__,
                  'rank',os.environ.get('LOCAL_RANK'),
                  'descriptor',desc,'graph_type',type(self.graphs[desc]).__name__,
                  'captured_graphs',len(self.graphs),flush=True)
        return result

    CudaGraphManager.run_fullgraph=audit_fullgraph

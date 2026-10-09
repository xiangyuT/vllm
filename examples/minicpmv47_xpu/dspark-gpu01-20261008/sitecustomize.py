import os

import vllm

vllm.__path__.append('/opt/venv/lib/python3.12/site-packages/vllm')
if os.environ.get('MINICPM_DSPARK_ATTENTION') == '1':
    import dspark_attention  # noqa: F401

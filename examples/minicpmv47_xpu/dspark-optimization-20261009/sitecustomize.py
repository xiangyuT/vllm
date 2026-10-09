import importlib.util
import os

path = "/workspace/preparation/dspark-iteration-20261009/sitecustomize.py"
spec = importlib.util.spec_from_file_location("minicpm_previous_startup", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if os.environ.get("MINICPM_ROUTING_AUDIT") == "1":
    import routing_audit

    routing_audit.install()

if os.environ.get("MINICPM_MARKOV_FP8") == "1":
    import markov_fp8

    markov_fp8.install()

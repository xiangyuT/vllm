import ast
from pathlib import Path

import regex

source=Path(__file__).resolve().parent.parent/'vllm-functional/tests/evals/gsm8k/gsm8k_eval.py'
tree=ast.parse(source.read_text())
nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)
       and node.name in ['get_answer_value','_build_gsm8k_prompts']]
namespace={'ast':ast,'re':regex,'INVALID':-9999999}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
get_answer_value=namespace['get_answer_value']

def build_prompts(train,test,num_questions,num_shots):
    namespace['load_gsm8k_data']=lambda:(train,test)
    return namespace['_build_gsm8k_prompts'](num_questions,num_shots,'')

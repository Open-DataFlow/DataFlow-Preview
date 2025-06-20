import sys
from dataflow.utils.registry import LazyLoader

cur_path = "dataflow/operators/eval/"
_import_structure = {  
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/eval/", _import_structure)

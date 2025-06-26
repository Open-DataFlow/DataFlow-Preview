from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval.GeneralText import NgramScorer

@OPERATOR_REGISTRY.register()
class NgramFilter(OperatorABC):

    def __init__(self, min_score=0.99, max_score=1, ngrams=5):
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = NgramScorer(ngrams)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        output_file = storage.write(filtered_dataframe)
        
        return [self.output_key]
        
        
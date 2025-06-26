import re
from tqdm import tqdm
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class NgramScorer(OperatorABC):
    
    def __init__(self, ngrams=5):
        self.logger = get_logger()
        self.ngrams = ngrams
    
    @staticmethod
    def get_desc(self, lang):
        return NotImplementedError("The description of NgramScorer is not implemented!")
    
    def _score_func(self, sample):
        content = sample 
        content = content.lower()
        content = re.sub(r'[^\w\s]', '', content)
        words = content.split()
        ngrams = [' '.join(words[i:i + self.ngrams]) for i in range(len(words) - (self.ngrams - 1))]
        unique_ngrams = set(ngrams)

        total_ngrams = len(ngrams)
        unique_ngrams_count = len(unique_ngrams)

        repetition_score = unique_ngrams_count / total_ngrams if total_ngrams > 0 else 0.0
        return repetition_score

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        scores = [self._score_func(sample) for sample in tqdm(dataframe[input_key], desc="NgramScorer Evaluating...")]
        return scores
    
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"NgramScore ready to evaluate, ")
        scores = self.eval(dataframe, input_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
        
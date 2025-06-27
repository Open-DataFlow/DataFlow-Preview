from typing import List
import re
from huggingface_hub import hf_hub_download
import fasttext
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
import numpy as np

@OPERATOR_REGISTRY.register()
class TextbookScorer(OperatorABC):
    def __init__(self, model_cache_dir=None):
        # Initialize model, tokenizer, and parameters
        model_path = hf_hub_download(
            repo_id='kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2',
            filename='model.bin',
            cache_dir=model_cache_dir
        )
        low_score=1.0
        mid_score=3.0
        high_score=5.0
        self.model = fasttext.load_model(model_path)
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'TextbookScore'

        # Mapping labels to scores
        self.score_dict = {
            '__label__Low': low_score,
            '__label__Mid': mid_score,
            '__label__High': high_score
        }

    @staticmethod
    def replace_newlines(text: str) -> str:
        """Replace newlines in the text with spaces."""
        return re.sub("\n+", " ", text)

    def _score_func(self, text_list: List[str]) -> List[float]:
        """Compute scores for a list of text samples."""
        # Replace newlines in text
        text_list = [self.replace_newlines(text) for text in text_list]
        
        # Predict the label and scores
        pred = self.model.predict(text_list, k=-1)
        
        score_list = []
        for labels, scores in zip(*pred):
            score = 0
            for label, score_value in zip(labels, scores):
                score += self.score_dict.get(label, 0) * score_value
            score_list.append(float(score))
        
        return score_list

    def eval(self, dataframe, input_key):
        """Evaluate the scores for each row in the dataframe."""
        scores = []
        text_list = dataframe[input_key]
        
        for sample in tqdm(text_list, desc="TextbookScorer Evaluating..."):
            score = self._score_func([sample])
            scores.append(score)
        
        return np.array(scores)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        """Read the dataframe, evaluate scores, and store results under the output_key."""
        dataframe = storage.read("dataframe")  # Read dataframe from storage
        scores = self.eval(dataframe, input_key, output_key)  # Evaluate the scores
        
        # Store the results under the output_key
        for i, score_list in enumerate(scores):
            dataframe[output_key] = score_list  # Assuming each score corresponds to a single output_key column
            
        storage.write(dataframe)  # Write the updated dataframe back to storage

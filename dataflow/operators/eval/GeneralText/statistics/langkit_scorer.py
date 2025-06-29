from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
import pandas as pd
from langkit import light_metrics, extract
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class LangkitScorer(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.llm_schema = light_metrics.init()

    @staticmethod
    def get_desc(self, lang):
        return NotImplementedError("The description of LangkitScorer is not implemented!")

    def _score_func(self, sample):
        # Process the sample using langkit
        df = pd.DataFrame({'prompt': [sample]})
        df['response'] = ''  
        enhanced_df = extract(df, schema=self.llm_schema)
        scores_dict = enhanced_df.to_dict(orient='records')[0]

        # Process the results to match the scoring format
        processed_scores = {}
        for k, v in scores_dict.items():
            if k == 'prompt':
                continue
            elif k.startswith('prompt.'):
                new_key = k[len('prompt.'):]  
                processed_scores[new_key] = v
            elif not (k == 'response' or k.startswith('response.')):
                processed_scores[k] = v  

        # Remove unwanted keys
        processed_scores.pop('has_patterns', None)

        # Create final result dictionary
        result = {}
        for k, v in processed_scores.items():
            score_key = f"Langkit{''.join([word.capitalize() for word in k.split('_')])}Score"
            result[score_key] = v

        return result

    def eval(self, dataframe, input_key):
        scores_list = []
        for sample in tqdm(dataframe[input_key], desc="LangkitScorer Evaluating..."):
            scores = self._score_func(sample)
            scores_list.append(scores)
        return scores_list
    
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info("LangkitScore ready to evaluate.")
        
        scores = self.eval(dataframe, input_key)
        # Flatten the nested dictionary of scores into the dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        storage.write(dataframe)

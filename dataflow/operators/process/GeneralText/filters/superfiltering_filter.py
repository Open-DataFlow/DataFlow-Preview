from dataflow.operators.eval.GeneralText import SuperfilteringScorer
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class SuperfilteringFilter(OperatorABC):

    def __init__(self, min_score=0.0, max_score=1.0, device='cuda', model_cache_dir='./dataflow_cache', max_length=512):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        self.scorer = SuperfilteringScorer(
            device=device,
            model_cache_dir=model_cache_dir,
            max_length=max_length
        )
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score={self.min_score} and max_score={self.max_score}...")

    @staticmethod
    def get_desc(self, lang):
        return "使用Superfiltering评分器过滤掉低质量数据" if lang == "zh" else "Filter out low-quality data using the Superfiltering scorer."

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_input_key: str = 'input', input_output_key: str = 'output', output_key: str = "superfiltering_filter_label"):
        self.input_instruction_key = input_instruction_key
        self.input_input_key = input_input_key
        self.input_response_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__ }...")

        # Get the scores for filtering
        scores = self.scorer.eval(dataframe, input_instruction_key, input_input_key, input_output_key)
        dataframe[self.output_key] = scores
        
        # Filter the dataframe based on the results
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

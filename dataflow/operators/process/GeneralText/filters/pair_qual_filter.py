from dataflow.operators.eval.GeneralText import PairQualScorer
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class PairQualFilter(OperatorABC):
    def __init__(self, min_score=2.5, max_score=10000, scorer_args: dict = None):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        # Initialize the scorer with provided arguments
        if scorer_args is None:
            scorer_args = {}
        self.scorer = PairQualScorer(**scorer_args)
        self.filter_name = 'PairQualFilter'
        
        # Log the initialization
        self.logger.info(f"Initializing {self.filter_name} with min_score={self.min_score}, max_score={self.max_score}...")

    @staticmethod
    def get_desc(self, lang):
        return "使用PairQual评分器过滤掉低质量数据" if lang == "zh" else "Filter out low-quality data using the PairQual scorer."

    def eval(self, dataframe, input_key):
        self.logger.info(f"Start evaluating {self.filter_name}...")
        
        # Get the scores using the scorer
        scores = self.scorer.eval(dataframe, input_key)

        # Return the scores for filtering
        return np.array(scores)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='pair_qual_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name}...")

        # Get the scores for filtering
        scores = self.eval(dataframe, self.input_key)

        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

from dataflow.operators.eval.GeneralText import QuratingScorer
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class QuratingFilter(OperatorABC):

    def __init__(self, min_scores: dict, max_scores: dict, scorer_args: dict = None):
        self.logger = get_logger()
        self.min_scores = min_scores
        self.max_scores = max_scores
        
        # Initialize the scorer with provided arguments
        if scorer_args is None:
            scorer_args = {}
        self.scorer = QuratingScorer(scorer_args)
        self.filter_name = 'QuratingFilter'
        self.logger.info(f"Initializing {self.filter_name} with min_scores={self.min_scores} and max_scores={self.max_scores}...")

    @staticmethod
    def get_desc(self, lang):
        return "使用Qurating评分器过滤掉低质量数据" if lang == "zh" else "Filter out low-quality data using the Qurating scorer."

    def eval(self, dataframe, input_key):
        self.logger.info(f"Start evaluating {self.filter_name}...")
        
        # Get the scores using the scorer
        _, scores = self.scorer(dataframe[input_key])

        # Return the scores for filtering
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name}...")

        # Get the scores for filtering
        scores = self.eval(dataframe, self.input_key)

        # Initialize results to all valid (1)
        results = np.ones(len(dataframe), dtype=int)

        # Iterate over each label to apply the filter
        for label in self.min_scores.keys():
            min_score = self.min_scores[label]
            max_score = self.max_scores[label]
            score_key = f"Qurating{''.join([word.capitalize() for word in label.split('_')])}Score"
            metric_scores = np.array(scores[score_key])
            
            # Apply score filter for the current label
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            results = results & metric_filter.astype(int)

        # Filter the dataframe based on the results
        filtered_dataframe = dataframe[results == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

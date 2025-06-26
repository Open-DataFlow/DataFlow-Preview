from dataflow.operators.eval.GeneralText import InstagScorer
from dataflow.core import OperatorABC
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class InstagFilter(OperatorABC):

    def __init__(self, min_score=0.0, max_score=1.0, scorer_args: dict = None):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        # Initialize the scorer
        if scorer_args is None:
            scorer_args = {}
        self.scorer = InstagScorer(scorer_args)
        self.filter_name = 'InstagFilter'
        self.logger.info(f"Initializing {self.filter_name} with min_score={self.min_score} and max_score={self.max_score}...")

    @staticmethod
    def get_desc(self, lang):
        return "使用Instag评分器过滤掉低标签数量数据" if lang == "zh" else "Filter out data with low tag counts using the Instag scorer."

    def eval(self, dataframe, input_key):
        self.logger.info(f"Start evaluating {self.filter_name}...")

        # Get the scores using the scorer
        _, scores = self.scorer(dataframe[input_key])

        # Extract the metric scores from the 'Default' key
        return np.array(scores['Default'])

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name}...")

        # Get the metric scores
        scores = self.eval(dataframe, self.input_key)

        # Filter records based on the score range
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

from dataflow.operators.eval.GeneralText import SuperfilteringScorer
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class SuperfilteringFilter(OperatorABC):

    def __init__(self, min_score=0.0, max_score=1.0, scorer_args: dict = None):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        # Initialize the scorer with provided arguments
        if scorer_args is None:
            scorer_args = {}
        self.scorer = SuperfilteringScorer(scorer_args)
        self.filter_name = 'SuperfilteringFilter'
        self.logger.info(f"Initializing {self.filter_name} with min_score={self.min_score} and max_score={self.max_score}...")

    @staticmethod
    def get_desc(self, lang):
        return "使用Superfiltering评分器过滤掉低质量数据" if lang == "zh" else "Filter out low-quality data using the Superfiltering scorer."

    def eval(self, dataframe, input_key):
        self.logger.info(f"Start evaluating {self.filter_name}...")
        
        # Get the scores using the scorer
        _, scores = self.scorer(dataframe[input_key])

        # Return the scores for filtering
        return np.array(scores['Default'])

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name}...")

        # Get the scores for filtering
        scores = self.eval(dataframe, self.input_key)

        # Apply the score filter for each record
        results = (self.min_score <= scores) & (scores <= self.max_score)

        # Filter the dataframe based on the results
        filtered_dataframe = dataframe[results]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

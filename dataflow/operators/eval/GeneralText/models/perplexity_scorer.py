from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
from dataflow import get_logger
from dataflow.operators.eval.GeneralText.models.Kenlm.model import KenlmModel

@OPERATOR_REGISTRY.register()
class PerplexityScorer(OperatorABC):
    def __init__(self, model_path, language, batch_size=1):
        # Initialize model and configuration
        self.model_path = model_path
        self.language = language
        self.batch_size = batch_size
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'PerplexityScore'
        self.model = KenlmModel.from_pretrained(self.model_path, self.language)
        self.logger = get_logger()

    def _score_func(self, text):
        # Calculate perplexity for a single text
        return self.model.get_perplexity(text)

    def eval(self, dataframe, input_key):
        # Evaluate perplexity for all texts in the dataframe
        scores = []
        for sample in tqdm(dataframe[input_key], desc="PerplexityScorer Evaluating..."):
            scores.append(self._score_func(sample))
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        # Read data, evaluate scores, and write the results
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[output_key] = scores
        storage.write(dataframe)

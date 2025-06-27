from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class PresidioScorer(OperatorABC):
    def __init__(self, model_cache_dir=None, language='en', device='cpu'):
        # Initialize parameters and load model
        self.language = language
        self.device = device
        self.model_cache_dir = model_cache_dir
        model_name = 'dslim/bert-base-NER'
        
        # Load tokenizer and model for NER
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=self.model_cache_dir).to(self.device)

        # Suppress warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="spacy_huggingface_pipelines")
        
        # Configure NLP engine
        model_config = [{
            "lang_code": self.language,
            "model_name": {
                "spacy": "en_core_web_sm",
                "transformers": model_name
            }
        }]
        
        self.nlp_engine = TransformersNlpEngine(models=model_config)
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine)
        
        # Scoring setup
        self.batch_size = 1
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'PresidioScore'
        self.logger = get_logger()

    def _score_func(self, text):
        """Calculate the PII count for a single text"""
        analysis_results = self.analyzer.analyze(text=text, language=self.language)
        pii_count = len(analysis_results)
        return pii_count

    def eval(self, dataframe, input_key):
        """Evaluate PII count for each text in the dataframe"""
        scores = []
        for sample in tqdm(dataframe[input_key], desc="PresidioScorer Evaluating..."):
            score = self._score_func(sample)
            scores.append(score)
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        """Read data, evaluate PII scores, and write the results back"""
        dataframe = storage.read("dataframe")  # Read dataframe from storage
        scores = self.eval(dataframe, input_key)  # Evaluate PII counts
        
        # Add results to dataframe and write back to storage
        dataframe[output_key] = scores
        storage.write(dataframe)

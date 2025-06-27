import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class FineWebEduScorer(OperatorABC):
    def __init__(self, model_name, model_cache_dir=None, device=None, batch_size=1):
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.model.eval()

        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'FineWebEduScore'

    def _score_func(self, sample):
        # Tokenize the input text
        tokenized_inputs = self.tokenizer(sample, return_tensors="pt", padding="longest", truncation=True).to(self.device)
        
        # Run the model
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy() 
        
        return logits.tolist()[0]  # Return as list for individual sample

    def eval(self, dataframe, input_key):
        scores = []
        for sample in tqdm(dataframe[input_key], desc="FineWebEduScorer Evaluating..."):
            score = self._score_func(sample)
            scores.append(score)
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        
        scores = self.eval(dataframe, input_key)
        
        # Write the results to the output key in the dataframe
        dataframe[self.output_key] = scores
        storage.write(dataframe)

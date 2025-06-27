from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
import torch
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class RMScorer(OperatorABC):
    def __init__(self, model_name, model_cache_dir=None, batch_size=1, device=None):
        # Initialize the model, tokenizer, and device
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'RewardModelScore'

        # Load the model and tokenizer
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)

    @staticmethod
    def get_desc(self, lang):
        return "使用RMScorer评估指令输出对模型的奖励得分" if lang == "zh" else "Evaluate instruction output based on reward-model-deberta-v3-large-v2."

    def _score_func(self, input_texts, output_texts):
        """Score a batch of text pairs."""
        inputs = self.tokenizer(input_texts, output_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Generate logits (model outputs)
        with torch.no_grad():
            logits = self.rank_model(**inputs).logits.cpu().detach().numpy()

        scores = logits.squeeze()

        if scores.ndim == 0:  # If it's a single value, convert it to a list
            scores = [float(scores)]

        return scores.tolist()

    def eval(self, dataframe, input_key, output_key):
        """Evaluate scores for all texts in the dataframe."""
        scores = []
        input_texts = dataframe[input_key]
        output_texts = dataframe[output_key]

        for sample_input, sample_output in tqdm(zip(input_texts, output_texts), desc="RMScorer Evaluating..."):
            score = self._score_func([sample_input], [sample_output])
            scores.append(score)

        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        """Read the dataframe, evaluate scores, and store the results."""
        dataframe = storage.read("dataframe")  # Read the dataframe from storage
        scores = self.eval(dataframe, input_key, output_key)  # Evaluate the scores

        # Store the results in the dataframe under the output_key
        for i, score_list in enumerate(scores):
            for j, score in enumerate(score_list):
                column_name = f"{output_key}_{j+1}"  # If multiple scores, store each in a separate column
                if column_name not in dataframe:
                    dataframe[column_name] = []
                dataframe[column_name].append(score)

        storage.write(dataframe)  # Write the updated dataframe back to storage

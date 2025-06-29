from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.special import softmax
from dataflow import get_logger
import torch
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class DeitaComplexityScorer(OperatorABC):
    def __init__(self, device=None, model_cache_dir=None, max_length=512):
        # Initialize model, tokenizer, and device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "hkust-nlp/deita-complexity-scorer"
        self.model_cache_dir = model_cache_dir
        self.max_length = max_length
        self.logger = get_logger()
        self.logger.info(f"Using local model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)

    @staticmethod
    def get_desc(self, lang):
        return "使用Deita指令复杂度分类器评估指令复杂度" if lang == "zh" else "Evaluate instruction complexity using the Deita instruction complexity classifier."

    def infer_complexity(self, input_text):
        # Format the input for the model
        complexity_template = ("You are a helpful assistant. Please identify the complexity score of the following user query. \n##Query: {instruction}\n##Complexity: ")
        user_input = complexity_template.format(instruction=input_text)
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)

        # Generate the output and calculate the complexity score
        outputs = self.model.generate(input_ids, max_new_tokens=self.max_length, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
        logprobs_list = outputs.scores[0][0]

        # Mapping of token IDs to complexity scores
        id2score = {
            29896: 1,  # Complexity level 1
            29906: 2,  # Complexity level 2
            29941: 3,  # Complexity level 3
            29946: 4,  # Complexity level 4
            29945: 5,  # Complexity level 5
            29953: 6   # Complexity level 6
        }

        score_template = np.array([1, 2, 3, 4, 5, 6])  # Define the score template
        score_logits = []

        # Collect the logits for the corresponding token IDs
        for k in id2score:
            score_logits.append(logprobs_list[k].cpu().numpy())

        score_logits = np.array(score_logits)
        score_npy = softmax(score_logits, axis=0)  # Apply softmax to get probabilities
        score_npy = score_npy * score_template  # Weight the scores by the corresponding complexity level
        final_score = np.sum(score_npy, axis=0)  # Sum the weighted scores to get the final score
        return final_score

    def eval(self, dataframe, input_instruction_key: str = 'instruction', input_output_key: str = 'output'):
        # Evaluate the quality score for each row in the dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        scores = []
        for sample in tqdm(dataframe[[input_instruction_key, input_output_key]].to_dict(orient='records'), desc="DeitaQualityScorer Evaluating..."):
            quality_score = self.infer_complexity(sample[input_instruction_key])  # assuming response and instruction are the same for now
            scores.append(quality_score)
        del self.tokenizer
        del self.model
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
        # Return as multiple columns
        return scores

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key: str = 'output', output_key: str = 'deita_complexity_score'):
        # Read the dataframe, evaluate scores, and store results
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_instruction_key, input_output_key)
        
        # Flatten results and write them to output_key in the dataframe
        dataframe[output_key] = scores        
        storage.write(dataframe)

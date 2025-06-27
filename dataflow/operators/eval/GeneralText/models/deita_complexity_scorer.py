from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.special import softmax
import torch
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class DeitaComplexityScorer(OperatorABC):
    def __init__(self, model_name="hkust-nlp/deita-complexity-scorer", model_cache_dir=None, device=None, max_length=512):
        # Initialize model, tokenizer, and device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.max_length = max_length
        self.batch_size = 1
        self.score_type = float  
        self.data_type = 'text'  
        self.score_name = 'DeitaComplexityScore'  
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)

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

    def eval(self, dataframe, input_key):
        # Evaluate the complexity score for each text in the dataframe
        scores = []
        for sample in tqdm(dataframe[input_key], desc="DeitaComplexityScorer Evaluating..."):
            complexity_score = self.infer_complexity(sample)
            scores.append(complexity_score)
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        # Read the data, compute complexity scores, and write the results
        dataframe = storage.read("dataframe")  # Read dataframe from storage
        scores = self.eval(dataframe, input_key)  # Evaluate the complexity scores
        
        # Flatten the nested result and write it to the output_key in the dataframe
        for i, score in enumerate(scores):
            for j, value in enumerate(score):
                column_name = f"{output_key}_{j+1}"  # Name each complexity level as separate columns
                if column_name not in dataframe:
                    dataframe[column_name] = []
                dataframe[column_name].append(value)

        storage.write(dataframe)  # Write the updated dataframe back to storage

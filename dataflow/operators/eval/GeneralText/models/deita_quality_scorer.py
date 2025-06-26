from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.special import softmax
import requests
import torch
from dataflow import get_logger
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class DeitaQualityScorer(OperatorABC):
    def __init__(self, model_name=None, model_cache_dir=None, device=None, max_length=10, use_API=False, api_url=None, api_model_name=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.use_API = use_API
        self.api_url = api_url or 'http://localhost:8000'
        self.api_model_name = api_model_name
        self.max_length = max_length
        self.batch_size = 1
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'DeitaQualityScore'
        self.logger = get_logger()

        if self.use_API:
            self.logger.info(f"Using API mode with model: {self.api_model_name}")
        else:
            self.logger.info(f"Using local model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)

        # Define token strings for quality scoring
        self.token_strs = ["1", "2", "3", "4", "5", "6"]
        self.score_template = np.array([1, 2, 3, 4, 5, 6])

    @staticmethod
    def get_desc(self, lang):
        return "使用Deita指令质量分类器评估指令质量" if lang == "zh" else "Evaluate instruction quality using the Deita instruction quality classifier."

    def infer_quality(self, input_text, resp_text):
        # Define the template and input format
        quality_template = ("You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question.\n"
                            "#Question#:\n{instruction}\n#Response#:\n{output}\n##Quality: ")
        user_input = quality_template.format(instruction=input_text, output=resp_text)

        if self.use_API:
            # API mode
            payload = {
                "model": self.api_model_name,
                "messages": [
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": self.max_length,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 6
            }

            response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()

            logprobs_list = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

            score_logits = []
            for token_str in self.token_strs:
                logprob = next((entry["logprob"] for entry in logprobs_list if entry["token"].strip() == token_str), -100)
                score_logits.append(logprob)

            score_logits = np.array(score_logits)
            score_npy = softmax(score_logits, axis=0)
            score_npy = score_npy * self.score_template
            final_score = np.sum(score_npy, axis=0)
            return final_score

        else:
            # Local inference mode
            input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids, max_new_tokens=self.max_length, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
            logprobs_list = outputs.scores[0][0]

            id2score = {
                29896: "1",
                29906: "2",
                29941: "3",
                29946: "4",
                29945: "5",
                29953: "6"
            }

            score_logits = []
            for k in id2score:
                score_logits.append(logprobs_list[k].cpu().numpy())

            score_logits = np.array(score_logits)
            score_npy = softmax(score_logits, axis=0)
            score_npy = score_npy * self.score_template
            final_score = np.sum(score_npy, axis=0)
            return final_score

    def eval(self, dataframe, input_key, output_key):
        # Evaluate the quality score for each row in the dataframe
        scores = []
        for sample in tqdm(dataframe[input_key], desc="DeitaQualityScorer Evaluating..."):
            quality_score = self.infer_quality(sample, sample)  # assuming response and instruction are the same for now
            scores.append(quality_score)
        
        # Return as multiple columns
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        # Read the dataframe, evaluate scores, and store results
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key, output_key)
        
        # Flatten results and write them to output_key in the dataframe
        for score_dict in scores:
            for i, value in enumerate(score_dict):
                column_name = f"{output_key}_{i+1}"  # Store each score in a separate column
                if column_name not in dataframe:
                    dataframe[column_name] = []
                dataframe[column_name].append(value)
        
        storage.write(dataframe)

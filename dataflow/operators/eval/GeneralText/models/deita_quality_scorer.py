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
    def __init__(self, device='cuda', model_cache_dir='', max_length=512):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'hkust-nlp/deita-quality-scorer'
        self.model_cache_dir = model_cache_dir
        self.max_length = max_length
        self.logger = get_logger()
        self.logger.info(f"Using local model: {self.model_name}")
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

    def eval(self, dataframe, input_instruction_key: str = 'instruction', input_output_key: str = 'output'):
        # Evaluate the quality score for each row in the dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        scores = []
        for sample in tqdm(dataframe[[input_instruction_key, input_output_key]].to_dict(orient='records'), desc="DeitaQualityScorer Evaluating..."):
            quality_score = self.infer_quality(sample[input_instruction_key], sample[input_output_key])  # assuming response and instruction are the same for now
            scores.append(quality_score)
        del self.tokenizer
        del self.model
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
        # Return as multiple columns
        return scores

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key: str = 'output', output_key: str = 'deita_quality_score'):
        # Read the dataframe, evaluate scores, and store results
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_instruction_key, input_output_key)
        
        # Flatten results and write them to output_key in the dataframe
        dataframe[output_key] = scores        
        storage.write(dataframe)

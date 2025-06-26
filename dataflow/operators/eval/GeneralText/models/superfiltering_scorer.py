from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval.GeneralText.models.Superfiltering.data_analysis import get_perplexity_and_embedding_whole_text, get_perplexity_and_embedding_part_text
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from dataflow.utils.storage import DataFlowStorage
import requests
import json
from math import exp
from dataflow.utils.utils import get_logger

# Superfiltering instruction quality (ifd) evaluation
# cited from: Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning
@OPERATOR_REGISTRY.register()
class SuperfilteringScorer(OperatorABC):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = args_dict.get('model_name')
        self.model_cache_dir = args_dict.get('model_cache_dir')
        self.use_API = args_dict.get('use_API', False)  # 默认使用本地
        self.api_url = args_dict.get('API_url', 'http://localhost:8000')
        self.api_model_name = args_dict.get('API_model_name')  # 使用vLLM时需要指定
        self.prompt = args_dict.get('prompt', 'none')
        self.max_length = args_dict.get('max_length', 512) 
        self.batch_size = 1
        self.score_type = float 
        self.data_type = 'text' 
        self.score_name = 'SuperfilteringScore' 
        self.logger = get_logger()
        
        if self.use_API:
            self.logger.info(f"Using API mode with model: {self.api_model_name}")
        else:
            self.logger.info(f"Using local model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map=self.device, 
                cache_dir=self.model_cache_dir, 
                output_hidden_states=True
            ).to(self.device)
    
    @staticmethod
    def get_desc(self, lang):
        return "使用Superfiltering评分器评估指令质量" if lang == "zh" else "Evaluate instruction quality using the Superfiltering scorer."

    def get_perplexity_api(self, text):
        """
        使用vLLM API获取文本的perplexity
        
        Args:
            text: 输入文本
            max_tokens: 生成的最大token数，默认为0（只计算输入文本的perplexity）
            
        Returns:
            perplexity值
        """
        try:
            # 使用completions API获取logprobs
            payload = {
                "model": self.api_model_name,
                "prompt": text,
                "max_tokens": 0,
                "echo": True,  # 确保返回输入文本的logprobs
                "logprobs": True  # 请求logprobs
            }
            
            response = requests.post(f"{self.api_url}/v1/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 从响应中提取logprobs
            logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
            
            # 忽略第一个token的logprob（通常是BOS token）
            if logprobs and len(logprobs) > 0 and logprobs[0] is None:
                logprobs = logprobs[1:]
                
            # 计算平均负对数似然（负对数似然越低，perplexity越低）
            if not logprobs or len(logprobs) == 0:
                return 0.0
                
            avg_neg_log_likelihood = -sum(logprobs) / len(logprobs)
            perplexity = exp(avg_neg_log_likelihood)
            return perplexity
            
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return 0.0

    def get_conditional_perplexity_api(self, whole_text, output_text):
        """
        使用API计算条件perplexity
        
        Args:
            whole_text: 完整文本（instruction + output）
            output_text: 只包含输出部分的文本
            
        Returns:
            条件perplexity值
        """
        try:
            # 计算整个文本的token数
            payload = {
                "model": self.api_model_name,
                "prompt": whole_text,
                "max_tokens": 0,
                "echo": True,
                "logprobs": True
            }
            
            response = requests.post(f"{self.api_url}/v1/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            
            token_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
            
            start_index = whole_text.rfind(output_text)
            payload = {
                "model": self.api_model_name,
                "prompt": whole_text[:start_index],
            }
            response = requests.post(f"{self.api_url}/tokenize", json=payload)
            response.raise_for_status()
            result = response.json()
            start_token = result['count']
            token_logprobs = token_logprobs[start_token:]
            
            avg_neg_log_likelihood = -sum(token_logprobs) / len(token_logprobs)
            perplexity = exp(avg_neg_log_likelihood)
            return perplexity
            
        except Exception as e:
            self.logger.error(f"API conditional perplexity calculation failed: {e}")
            return 0.0

    def inference(self, instruction, input_text, output):
        """
        根据use_API标志决定使用API或本地模型进行推理
        
        Args:
            instruction: 指令文本
            input_text: 输入文本
            output: 输出文本
            
        Returns:
            超滤得分(superfiltering score)
        """
        PROMPT_DICT_NONE = {
            "prompt_input": (
                "{instruction}\n{input}\n"
            ),
            "prompt_no_input": (
                "{instruction}\n"
            ),
        }

        if self.prompt == 'none':
            prompt_no_input = PROMPT_DICT_NONE["prompt_no_input"]
            prompt_input = PROMPT_DICT_NONE["prompt_input"]

        if input_text == '':
            temp_dict = {'instruction': instruction}
            prompt_to_use = prompt_no_input.format_map(temp_dict)
            whole_text = prompt_to_use + output
            instruction = prompt_to_use
        else:
            temp_dict = {'instruction': instruction, 'input': input_text}
            prompt_to_use = prompt_input.format_map(temp_dict)
            whole_text = prompt_to_use + output
            instruction = prompt_to_use

        if output == '':
            return None
            
        if self.use_API:
            # 使用API计算perplexity
            ppl_out_alone = self.get_perplexity_api(output)
            ppl_out_condition = self.get_conditional_perplexity_api(whole_text, output)
            
            if ppl_out_alone != 0:
                score = ppl_out_condition / ppl_out_alone
            else:
                score = 0
        else:
            # 使用本地模型计算perplexity
            instruction_input_ids = self.tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            instruction_len = instruction_input_ids.shape[1]
            
            ppl_out_alone, _ = get_perplexity_and_embedding_whole_text(self.tokenizer, self.model, output, self.max_length - instruction_len + 1, self.device)
            ppl_out_condition, _ = get_perplexity_and_embedding_part_text(self.tokenizer, self.model, whole_text, output, self.max_length, self.device)

            if ppl_out_alone != 0:
                score = ppl_out_condition / ppl_out_alone
            else:
                score = 0

        if score != score:  # 检查NaN
            score = None
            
        return score

    def evaluate_batch(self, batch):
        instruction = batch.get('instruction', [''])[0]
        output = batch.get('output', [''])[0]
        input_text = batch.get('input', [''])[0] if 'input' in batch else ''
        
        if not output:
            score = None
        else:
            score = self.inference(instruction, input_text, output)
            
        return [score]
    
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

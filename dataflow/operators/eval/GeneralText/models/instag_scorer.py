from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
import requests
import torch
from dataflow import get_logger
import json

@OPERATOR_REGISTRY.register()
class InstagScorer(OperatorABC):
    def __init__(self, model_name='', model_cache_dir='', device='cuda', max_new_tokens=1024, use_API=False, api_url='http://0.0.0.0:8003', api_model_name='', temperature=0, do_sample=False, num_return_sequences=1, return_dict_in_generate=True):
        # Initialize parameters and model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.use_API = use_API
        self.api_url = api_url
        self.api_model_name = api_model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences
        self.return_dict_in_generate = return_dict_in_generate
        self.batch_size = 1
        self.score_type = float
        self.logger = get_logger()

        if self.use_API:
            self.logger.info(f"Using API mode with URL: {self.api_url}, model: {self.api_model_name}")
        else:
            self.logger.info(f"Using local model: {self.model_name}")
        # Token strings and score template
        self.token_strs = ["1", "2", "3", "4", "5", "6"]
        self.score_template = np.array([1, 2, 3, 4, 5, 6])

    @staticmethod
    def get_desc(self, lang):
        return "使用Instag评分器评估指令意图标签" if lang == "zh" else "Evaluate instruction intention tags using the Instag scorer."

    def make_prompt(self, query):
        prompt = f"Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{\"tag\": str, \"explanation\": str}}.\nUser query: {query}"
        messages = [("user", prompt), ("Assistant", None)]
        seps = [" ", "</s>"]
        ret = "system: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions." + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def format_api_messages(self, query):
        """Format the messages for the API call."""
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        user_content = (
            f"Please identify tags of user intentions in the following user query and provide an explanation for each tag. "
            f"Please respond in the JSON format {{'tag': str, 'explanation': str}}.\n"
            f"User query: {query}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return messages

    def inference_batch(self, queries):
        """Process batch of queries using either local model or API."""
        if self.use_API:
            json_outputs = []
            for query in queries:
                messages = self.format_api_messages(query)
                payload = {
                    "model": self.api_model_name,
                    "messages": messages,
                    "max_tokens": self.max_new_tokens,
                    "temperature": self.temperature
                }

                try:
                    self.logger.info(f"Calling API: {self.api_url}/v1/chat/completions, query: {query[:30]}...")
                    response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload, timeout=60)
                    response.raise_for_status()
                    result = response.json()

                    self.logger.info(f"API response: {str(result)[:200]}...")

                    if result["choices"] and result["choices"][0].get("message"):
                        generated_text = result["choices"][0]["message"].get("content", "")
                        self.logger.info(f"Generated text: {generated_text[:100]}...")
                        
                        string_output = generated_text.strip()
                        try:
                            json_output = json.loads(string_output)
                            self.logger.info(f"Parsed JSON: {json_output}")
                        except json.JSONDecodeError:
                            self.logger.warning(f"JSON parse error: {string_output}")
                            json_output = {"tag": "Parsing error", "explanation": string_output[:100]}

                        json_outputs.append(json_output)
                    else:
                        self.logger.warning(f"API response format error: {result}")
                        json_outputs.append({"tag": "API response format error", "explanation": str(result)[:100]})

                except requests.exceptions.RequestException as e:
                    self.logger.error(f"API request error: {e}")
                    json_outputs.append({"tag": "API call error", "explanation": str(e)})
            
            return json_outputs
        else:
            input_strs = [self.make_prompt(query) for query in queries]
            input_tokens = self.tokenizer(input_strs, return_tensors="pt", padding=True)

            if torch.cuda.is_available():
                input_tokens = {key: value.to(self.device) for key, value in input_tokens.items()}

            output = self.model.generate(
                input_tokens['input_ids'],
                temperature=self.temperature,
                do_sample=self.do_sample,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_return_sequences,
                return_dict_in_generate=self.return_dict_in_generate,
            )
            
            num_input_tokens = input_tokens["input_ids"].shape[1]
            output_tokens = output.sequences
            generated_tokens = output_tokens[:, num_input_tokens:]
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            json_outputs = []
            for generated_text in generated_texts:
                string_output = generated_text.strip()
                try:
                    json_output = json.loads(string_output)
                except json.JSONDecodeError:
                    self.logger.warning(f"JSON parse error: {string_output}")
                    json_output = {"tag": "Parsing error", "explanation": string_output[:100]}
                json_outputs.append(json_output)
            
            return json_outputs

    def _score_func(self, sample, input_instruction_key):
        """Evaluate a batch of queries and return corresponding scores."""
        # query = sample.get(input_instruction_key, '')
        json_output = self.inference_batch([sample])[0]
        complexity_score = None
        if isinstance(json_output, list):
            complexity_score = len(json_output)
            self.logger.info(f"列表类型JSON,标签数量: {complexity_score}")
        elif isinstance(json_output, dict) and "tag" in json_output:  # 单个标签返回为字典
            complexity_score = 1
            self.logger.info(f"字典类型JSON,包含tag字段,评分为1")
        elif isinstance(json_output, dict) and len(json_output) > 0:  # 其他字典类型,有内容
            complexity_score = 1
            self.logger.info(f"其他字典类型JSON,评分为1: {json_output}")
        else:
            complexity_score = 0
            self.logger.warning(f"未识别的JSON类型或空数据,评分为0: {json_output}") 
        return complexity_score

    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str):
        if not self.use_API:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
            self.model.requires_grad_(False)
            self.model.eval()
        scores = []
        for sample in tqdm(dataframe[input_instruction_key], desc="InstagScorer Evaluating..."):
            scores.append(self._score_func(sample, input_instruction_key))
        if not self.use_API:
            del self.tokenizer
            del self.model
            import gc;
            gc.collect()
            torch.cuda.empty_cache()
        return scores

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', output_key: str = 'instag_score'):
        """Process the batch and store results under the specified output_key."""
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_instruction_key)
        
        # Store the results in the output_key (create multiple columns if needed)
        dataframe[output_key] = scores
        
        storage.write(dataframe)

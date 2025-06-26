import torch
from torch import nn
import transformers
import requests
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
import json
from dataflow.utils.utils import get_logger
import os

class BertForRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.regression = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        encoded = self.bert(**inputs)
        score = self.regression(encoded['pooler_output'])
        return encoded, score


@OPERATOR_REGISTRY.register()
class PairQualScorer(OperatorABC):
    def __init__(self, model_cache_dir:str=None, device="cuda", batch_size=64, use_API=False, API_url=None, API_model_name=None, model_state_dict=None, max_length=512):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_cache_dir = model_cache_dir
        self.use_API = use_API
        self.API_url = API_url
        self.API_model_name = API_model_name
        self.max_length = max_length
        self.model_state_dict = model_state_dict
        self.batch_size = batch_size
        self.score_type = float
        self.score_name = 'PairQualScorer'
        self.logger = get_logger()

        # Hardcoding model name or get it from another source
        self.model_name = "bert-base-uncased"  # Hardcoded model name

        self.logger.info(f"Initializing tokenizer with model_name: {self.model_name}")
        try:
            if self.model_cache_dir and os.path.exists(self.model_cache_dir):
                print(self.model_cache_dir)
                self.tokenizer = transformers.BertTokenizerFast.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
                self.logger.info(f"Tokenizer initialized successfully from cache for {self.model_name}.")
            else:
                # If model_cache_dir doesn't exist or is not provided, fall back to downloading from Hugging Face Hub
                self.tokenizer = transformers.BertTokenizerFast.from_pretrained(self.model_name)
                self.logger.info(f"Tokenizer initialized successfully from Hugging Face for {self.model_name}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer for {self.model_name}: {e}")
            raise e


        if self.use_API:
            self.logger.info(f"Using API mode with URL: {self.API_url}, model: {self.API_model_name}")
            if not self.API_model_name:
                self.logger.error("API_model_name is not configured for API mode. This will cause errors.")
        else:
            self.logger.info(f"Using local model: {self.model_name}")
            self.model = BertForRegression(self.model_name)
            if self.model_state_dict:
                self.model.load_state_dict(torch.load(self.model_state_dict, map_location='cpu'))
            self.model.to(self.device).eval()

    @staticmethod
    def get_desc(self, lang):
        return "使用PairQual评分器评估文本质量" if lang == "zh" else "Evaluate text quality using the PairQual scorer."

    def get_embeddings_api(self, texts):
        """通过API获取文本嵌入"""
        try:
            payload = {"model": self.API_model_name, "input": texts}
            payload_str = json.dumps(payload)
            self.logger.info(f"Sending API request to {self.API_url}/v1/embeddings with payload: {payload_str[:500]}...")

            response = requests.post(f"{self.API_url}/v1/embeddings", json=payload, timeout=60)
            self.logger.info(f"API response status code: {response.status_code}")
            response.raise_for_status()

            result = response.json()
            embeddings = [data["embedding"] for data in result["data"]]
            return np.array(embeddings)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None

    def inference(self, input_text):
        """根据use_API标志决定使用API或本地模型进行推理"""
        if self.use_API:
            if not isinstance(input_text, str):
                self.logger.warning(f"Invalid input for API inference. Skipping embedding.")
                return 0.0
            input_text_to_embed = input_text
            tokenized_output = self.tokenizer(input_text, truncation=True, max_length=self.max_length, return_attention_mask=False, return_token_type_ids=False)
            input_text_to_embed = self.tokenizer.decode(tokenized_output['input_ids'], skip_special_tokens=True)
            embeddings = self.get_embeddings_api([input_text_to_embed])
            if embeddings is None or embeddings.size == 0:
                return 0.0
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                score = self.model.regression(embedding_tensor)
            return score.item()
        else:
            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                _, score = self.model(inputs)
            return score.item()

    def eval(self, dataframe, input_key):
        """批量评估"""
        scores = []
        for sample in tqdm(dataframe[input_key], desc="PairQualScorer Evaluating..."):
            score = self.inference(sample)
            scores.append(score)
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        """读取数据并运行评分"""
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[output_key] = scores
        storage.write(dataframe)

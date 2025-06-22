from tqdm import tqdm
import pandas as pd
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.Storage import FileStorage
from dataflow.utils.Operator import Operator
from transformers import AutoTokenizer


@OPERATOR_REGISTRY.register()
class AnswerTokenLengthFilter(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.max_answer_token_length = self.config['max_answer_token_length']
        self.tokenizer_dir = self.config['tokenizer_dir']

        self.logger = get_logger()
        self.datastorage = FileStorage(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'input_key', 'max_answer_token_length', 'tokenizer_dir']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子根据token数量过滤过长的答案。\n\n"
                "输入参数：\n"
                "- max_answer_token_length：最大token数\n"
                "- tokenizer_dir：分词器路径\n"
                "- read_min/max_score：分数范围\n\n"
                "输出参数：\n"
                "- 长度合规返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "Filters answers exceeding specified token length limit.\n\n"
                "Input Parameters:\n"
                "- max_answer_token_length: Maximum allowed tokens\n"
                "- tokenizer_dir: Tokenizer directory\n"
                "- read_min/max_score: Score range\n\n"
                "Output Parameters:\n"
                "- Returns 1 if within limit, 0 otherwise"
            )
        else:
            return "AnswerTokenLengthFilter enforces answer length constraints"


    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = []

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")
        missing_keys = [key for key in required_keys if key not in dataframe.columns]

        if missing_keys:
            raise ValueError(f"The following required columns are missing from the dataframe: {missing_keys}")

    # def run(self, dataset):
    #     def get_token_count(input_string):
    #         tokens = self.tokenizer.encode(input_string, add_special_tokens=False)
    #         return len(tokens)

    #     return np.array([get_token_count(item[self.keys]) <= self.max_answer_token_length for item in dataset]).astype(int)

    def run(self):
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        self.logger.info(f"Found {len(dataframe)} rows in the dataframe")
        self._validate_dataframe(dataframe)

        def get_token_count(input_string):
            tokens = self.tokenizer.encode(input_string, add_special_tokens=False)
            return len(tokens)

        output = []
        for i, text in tqdm(enumerate(dataframe[self.input_key]), desc="Checking token lengths"):
            is_valid = get_token_count(text) <= self.max_answer_token_length
            if is_valid:
                output.append(dataframe.iloc[i])
        
        dataframe = pd.DataFrame(output)

        self.datastorage.write(self.output_file, dataframe)
        self.logger.info(f"Saved {len(dataframe)} filtered rows to {self.output_file}")
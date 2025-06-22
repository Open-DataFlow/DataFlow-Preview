from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger

from dataflow.utils.Storage import FileStorage
from dataflow.utils.Operator import Operator
@OPERATOR_REGISTRY.register()
class AnswerNgramFilter(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.question_key = self.config['question_key']
        self.answer_key = self.config['answer_key']
        self.min_score = self.config['min_score']
        self.max_score = self.config['max_score']
        self.logger = get_logger()
        self.datastorage = FileStorage(config)
        self.ngrams = self.config['ngrams']

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'question_key', 'answer_key', 'min_score', 'max_score']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子基于n-gram重复率过滤答案，检测回答中的重复模式。\n\n"
                "输入参数：\n"
                "- min_score：最小可接受分数\n"
                "- max_score：最大可接受分数\n"
                "- ngrams：n-gram大小\n\n"
                "输出参数：\n"
                "- 分数在范围内返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "This filter detects repetitive patterns using n-gram repetition scores.\n\n"
                "Input Parameters:\n"
                "- min_score: Minimum acceptable score\n"
                "- max_score: Maximum acceptable score\n"
                "- ngrams: Size of n-grams\n\n"
                "Output Parameters:\n"
                "- Returns 1 if score is within range, 0 otherwise"
            )
        else:
            return "AnswerNgramFilter detects answer repetition"
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.question_key, self.answer_key]
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
        
    def run(self):
        '''
        
        '''
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        self.logger.info(f"Found {len(dataframe)} rows in the dataframe")

        scores = []
        for sample in dataframe.itertuples(index=False):
            try:
                answer = getattr(sample, self.question_key)
                answer += getattr(sample, self.answer_key, "")
            except AttributeError:
                answer = getattr(sample, self.question_key)

            content = answer.lower()
            content = re.sub(r'[^\w\s]', '', content)
            words = content.split()
            ngrams = [' '.join(words[i:i + self.ngrams]) for i in range(len(words) - (self.ngrams - 1))]
            unique_ngrams = set(ngrams)

            total_ngrams = len(ngrams)
            unique_ngrams_count = len(unique_ngrams)

            repetition_score = unique_ngrams_count / total_ngrams if total_ngrams > 0 else 0.0
            scores.append(repetition_score)

        indexes = np.array([self.min_score <= s <= self.max_score for s in scores])
        dataframe = dataframe[indexes]

        self.logger.info(f"Filtered down to {len(dataframe)} rows with repetition score in [{self.min_score}, {self.max_score}]")
        self.datastorage.write(self.output_file, dataframe)
        self.logger.info(f"Results saved to {self.output_file}")
from dataflow.utils.reasoning_utils.Prompts import QuestionDifficultyPrompt
import pandas as pd
import json
import os
import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger

from dataflow.utils.Storage import FileStorage
from dataflow.utils.Operator import Operator
from dataflow.utils.utils import init_model

@OPERATOR_REGISTRY.register()
class QuestionDifficultyClassifier(Operator):
    def __init__(self, config):
        """
        Initialize the QuestionCategoryClassifier with the provided configuration.
        """
        self.check_config(config)
        self.config = config
        self.prompts = QuestionDifficultyPrompt()
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.output_key = self.config.get("output_key", "difficulty_score")
        self.logger = get_logger()
        
        self.generator = init_model(config)
        self.datastorage = FileStorage(config)
    
    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'generator_type']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
    
    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于评估问题的难度等级。"
                "通过大语言模型分析问题复杂度，输出1-10级的难度评分。\n\n"
                "输入参数：\n"
                "- eval_stage：评估阶段标识\n"
                "- read_min/max_score：分数过滤阈值\n"
                "- 其他参数同QuestionCategoryClassifier\n\n"
                "输出参数：\n"
                "- difficulty_score：数值型难度评分（1-10）"
            )
        elif lang == "en":
            return (
                "Evaluates question difficulty level using LLM analysis. "
                "Outputs numerical difficulty score from 1 to 10.\n\n"
                "Input Parameters:\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- read_min/max_score: Score filtering thresholds\n"
                "- Other params same as QuestionCategoryClassifier\n\n"
                "Output Parameters:\n"
                "- difficulty_score: Numerical difficulty rating (1-10)"
            )
        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

        
    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        formatted_prompts = []
        for i, text in enumerate(dataframe[self.input_key]):
            if text is not None:
                used_prompt = self.prompts.question_synthesis_prompt(text)
            else:
                used_prompt = None
            formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def run(self):
        """
        Run the question difficulty classification process.
        """
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.generator.generate_from_input(formatted_prompts)

        rating_scores = []
        for response in responses:
            match = re.search(r'Rating:\s*((\d+\.\d+)|\d+)', response)
            if match:
                score_str = match.group(1).rstrip('.')
                try:
                    score = float(score_str)
                except ValueError:
                    score = -1
            else:
                score = -1
            rating_scores.append(score)
        dataframe[self.output_key] = rating_scores
        
        self.datastorage.write(self.output_file, dataframe)
        self.logger.info(f"Classification results saved to {self.output_file}")
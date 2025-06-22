from dataflow.utils.reasoning_utils.Prompts import QuestionSynthesisPrompt
import pandas as pd
import random
import os
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger

from dataflow.utils.Storage import FileStorage
from dataflow.utils.Operator import Operator
from dataflow.utils.utils import init_model

@OPERATOR_REGISTRY.register()
class QuestionGenerator(Operator):
    def __init__(self, config):
        """
        Initialize the QuestionGenerator with the provided configuration.
        """
        self.check_config(config)
        self.config = config
        self.prompts = QuestionSynthesisPrompt()
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.num_prompts = self.config.get("num_prompts", 1)
        if self.num_prompts not in range(1,6):
            raise ValueError("num_prompts must be an integer between 1 and 5 (inclusive)")
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
                "该算子用于基于现有问题生成新问题。\n\n"
                "输入参数：\n"
                "- eval_stage：评估阶段标识\n"
                "- read_min/max_score：分数过滤阈值\n"
                "- 其他参数同基础分类器\n\n"
                "输出参数：\n"
                "- generated_questions：生成的新问题列表（每个原问题生成1-5个）"
            )
        elif lang == "en":
            return (
                "Generates new questions based on existing ones. "
                "Produces 1-5 new questions per original question.\n\n"
                "Input Parameters:\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- read_min/max_score: Score filtering thresholds\n"
                "- Other params same as base classifier\n\n"
                "Output Parameters:\n"
                "- generated_questions: List of newly generated questions"
            )
        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = ["Synth_or_Input"]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

        
    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions based on num_prompts.
        """
        diversity_mode = [
            "1, 2, 3",
            "1, 2, 4",
            "1, 2, 5",
            "1, 4, 5",
            "1, 2, 3, 4, 5"
        ]

        formatted_prompts = []
        for question in dataframe[self.input_key]:
            if self.num_prompts == 0:
                formatted_prompts.append("")  # Skip generating for this question
            else:
                # Randomly choose the required number of transformations from diversity_mode
                selected_items = random.sample(diversity_mode, self.num_prompts)
                for selected_item in selected_items:
                    used_prompt = self.prompts.question_synthesis_prompt(selected_item, question)
                    formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def run(self):
        """
        Run the question generation process.
        """
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.generator.generate_from_input(formatted_prompts)

        new_rows = pd.DataFrame({
            self.input_key: responses,
        })
        new_rows["Synth_or_Input"] = "synth"
        dataframe["Synth_or_Input"] = "input"
        
        dataframe = pd.concat([dataframe, new_rows], ignore_index=True)
        dataframe = dataframe[dataframe[self.input_key].notna()]
        dataframe = dataframe[dataframe[self.input_key] != ""]

        self.datastorage.write(self.output_file, dataframe)
        self.logger.info(f"Generated questions saved to {self.output_file}")
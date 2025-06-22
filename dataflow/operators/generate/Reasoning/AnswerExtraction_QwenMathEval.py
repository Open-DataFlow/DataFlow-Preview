import pandas as pd
from tqdm import tqdm
import logging
import re
from word2number import w2n
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils import Operator
from dataflow.utils.Storage import FileStorage
from dataflow.utils.reasoning_utils.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor

# The main class to manage the entire extraction process
@OPERATOR_REGISTRY.register()
class AnswerExtraction_QwenMathEval(Operator):
    """
    A class to handle the process of extracting answers from a dataset.
    """

    def __init__(self, config: dict):
        """
        Initializes the AnswerExtraction_QwenMathEval class.
        """
        self.check_config(config)
        self.config = config
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.response_key = self.config['response_key']
        self.extraction_key = self.config['extraction_key']
        self.data_name = self.config.get('dataset_name', None)
        self.logger = get_logger()
        self.datastorage = FileStorage(config)

        # Initialize helpers
        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        self.answer_extractor = AnswerExtractor(string_cleaner)

    def check_config(self):
        """
        Ensures that the configuration contains all necessary keys.
        Must have either (input_file and output_file) or db_name.
        """
        config = self.config  # for brevity

        # Check general required keys
        required_keys = ['response_key', 'extraction_key']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Key {key} is missing in the config")

        # Check data source requirement
        has_file_io = 'input_file' in config and 'output_file' in config
        if not has_file_io:
            raise ValueError("Config must contain both 'input_file' and 'output_file'")

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于从数学问题回答中提取规范化答案表达式，进行字符串清洗、单位处理和格式标准化。\n\n"
                "输入参数：\n"
                "- input_key：输入数据字段名\n"
                "- answer_key：原始答案字段名\n"
                "- output_key：处理后的答案字段名\n"
                "- unit_texts：需要过滤的单位文本列表\n\n"
                "输出参数：\n"
                "- output_key：标准化后的数学表达式字段"
            )
        elif lang == "en":
            return (
                "This operator extracts and normalizes mathematical expressions from answers, "
                "performing string cleaning, unit processing and format standardization.\n\n"
                "Input Parameters:\n"
                "- input_key: Input data field name\n"
                "- answer_key: Raw answer field name\n"
                "- output_key: Processed answer field name\n"
                "- unit_texts: List of unit texts to filter\n\n"
                "Output Parameters:\n"
                "- output_key: Standardized mathematical expression field"
            )
        else:
            return "AnswerExtraction_QwenMathEval performs mathematical answer normalization and standardization."

    def run(self):
        """
        Executes the answer extraction process.
        """
        raw_dataframe = self.datastorage.read(self.input_file, "dataframe")
        key_list = raw_dataframe.columns.to_list()
        if self.response_key not in key_list:
            raise ValueError(f"response_key: {self.response_key} not found in dataframe columns.")

        self.logger.info(f"Found {len(raw_dataframe)} rows.")
        extractions = [
            self.answer_extractor.extract_answer(resp, self.data_name)
            for resp in tqdm(raw_dataframe[self.response_key], desc='Processing')
        ]
        raw_dataframe[self.extraction_key] = extractions
        self.datastorage.write(self.output_file, raw_dataframe)

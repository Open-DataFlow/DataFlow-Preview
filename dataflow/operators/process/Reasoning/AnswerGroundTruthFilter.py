from math_verify import parse, verify
import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.reasoning_utils.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor
from dataflow.utils.Storage import FileStorage
from dataflow.utils.utils import get_logger
from dataflow.utils.Operator import Operator

@OPERATOR_REGISTRY.register()
class AnswerGroundTruthFilter(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.filter_name = 'AnswerGroundTruthFilter'
        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        self.answer_extractor = AnswerExtractor(string_cleaner)
        name2compare = {
            'exact': self.exact_compare,
            'math_verify': self.math_verify_compare
        }
        self.logger = get_logger()
        self.compare = name2compare[config.get('compare_method', 'exact')]
        self.datastorage = FileStorage(config)
        self.input_file = config['input_file']
        self.output_file = config['output_file']
        self.test_answer_key = config['test_answer_key']
        self.gt_answer_key = config['gt_answer_key']

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'test_answer_key', 'gt_answer_key']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    def exact_compare(self, answer, ground_truth):
        return str(answer) == str(ground_truth)
    
    def math_verify_compare(self, answer, ground_truth):
        try:
            return verify(parse(str(ground_truth)), parse(str(answer)))
        except:
            try:
                return verify(parse(ground_truth), parse(answer))
            except:
                return False

    def run(self):
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        output = []
        answers = dataframe[self.test_answer_key]
        ground_truths = dataframe[self.gt_answer_key]
        for i in range(len(answers)):
            final_answer =  self.answer_extractor.extract_answer(answers[i], None)
            if self.compare(final_answer, ground_truths[i]):
                output.append(dataframe.iloc[i])
        output = pd.DataFrame(output)
        self.datastorage.write(self.output_file, output)

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于对比预测答案与标准答案的匹配度，支持精确匹配和数学验证两种方式。\n\n"
                "输入参数：\n"
                "- test_answer_key：预测答案字段名\n"
                "- gt_answer_key：标准答案字段名\n"
                "- compare_method：比较方法（exact/math_verify）\n\n"
                "输出参数：\n"
                "- 匹配成功返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "This operator compares predicted answers against ground truth using exact or mathematical verification.\n\n"
                "Input Parameters:\n"
                "- test_answer_key: Predicted answer field\n"
                "- gt_answer_key: Ground truth field\n"
                "- compare_method: Comparison method (exact/math_verify)\n\n"
                "Output Parameters:\n"
                "- Returns 1 for matches, 0 otherwise"
            )
        else:
            return "AnswerGroundTruthFilter performs answer validation"
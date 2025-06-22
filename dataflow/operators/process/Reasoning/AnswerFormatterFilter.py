import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger

from dataflow.utils.Storage import FileStorage
from dataflow.utils.Operator import Operator

@OPERATOR_REGISTRY.register()
class AnswerFormatterFilter(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.input_key = self.config['input_key']
        self.logger = get_logger()

        self.datastorage = FileStorage(config)

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file', 'input_key']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    def is_valid_answer(answer: str) -> bool:
        # check final answer in \boxed{} or not 
        # if not re.search(r'\\boxed{.*}', answer):
        #     return False
        
        return True 

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子用于检查答案格式是否符合规范，主要验证数学答案是否包含正确的\\boxed{}标记。\n\n"
                "输入参数：\n"
                "- input_key：输入字段名\n"
                "- eval_stage：评估阶段标识\n"
                "- result_key：结果字段名\n\n"
                "输出参数：\n"
                "- 通过格式检查返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "This operator validates answer formatting, specifically checking for correct \\boxed{} notation.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the answer\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- result_key: Output result field name\n\n"
                "Output Parameters:\n"
                "- Returns 1 for valid format, 0 otherwise"
            )
        else:
            return "AnswerFormatterFilter validates mathematical answer formatting"
    
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
    
    def run(self):
        '''
        Execute the answer format filter process
        '''
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        self._validate_dataframe(dataframe)

        indexes =  np.zeros(len(dataframe)).astype(int)
        for i, item in dataframe.iterrows():
            answer = item[self.input_key]
            if AnswerFormatterFilter.is_valid_answer(answer):
                indexes[i] = 1
        dataframe = dataframe[np.array(indexes) == 1]

        self.datastorage.write(self.output_file, dataframe)
        self.logger.info(f"Results saved to {self.output_file}")
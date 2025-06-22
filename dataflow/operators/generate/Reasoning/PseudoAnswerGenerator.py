import sys
from collections import defaultdict, Counter
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.Storage import FileStorage
from dataflow.utils.Operator import Operator
from dataflow.utils.reasoning_utils.Prompts import AnswerGeneratorPrompt
from dataflow.utils.utils import init_model
from dataflow.utils.reasoning_utils.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor

@OPERATOR_REGISTRY.register()
class PseudoAnswerGenerator(Operator):
    '''
    Pseudo Answer Generator is a class that generates answers for given questions, then choose the most frequent answer.
    '''
    def __init__(self,config: dict):
        self.config = config
        self.prompt = AnswerGeneratorPrompt()
        self.datastorage = FileStorage(config)
        self.input_file = self.config["input_file"]
        self.output_file = self.config["output_file"]
        self.input_key = self.config["input_key"]
        self.output_key_answer = self.config["output_key_answer"]
        self.output_key_answer_value = self.config["output_key_answer_value"]
        self.output_key_solutions = self.config["output_key_solutions"]
        self.output_key_correct_solution_example = self.config["output_key_correct_solution_example"]
        self.max_times = self.config["max_times"]
        self.logger = get_logger()
        self.model = init_model(self.config)
        self.extractor = self.get_extractor()
    def check_config(self):
        required_keys = ["input_file", "output_file", "input_key", "output_key_answer", "output_key_answer_value", "output_key_solutions", "output_key_correct_solution_example", "max_times"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
    
    def get_extractor(self):
        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        answer_extractor = AnswerExtractor(string_cleaner)
        return answer_extractor
        
    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "该算子生成多个候选答案并通过统计选择最优解，实现伪答案生成。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- output_file：输出文件路径\n"
                "- max_times：最大生成次数\n"
                "- selection_mode：统计选择模式（frequency/consistency）\n\n"
                "输出参数：\n"
                "- final_answer：最终选择答案字段\n"
                "- candidate_answers：候选答案列表字段"
            )
        elif lang == "en":
            return (
                "This operator generates multiple candidate answers and selects the optimal solution "
                "through statistical analysis.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- output_file: Output file path\n"
                "- max_times: Maximum generation times\n"
                "- selection_mode: Statistical selection mode (frequency/consistency)\n\n"
                "Output Parameters:\n"
                "- final_answer: Selected answer field\n"
                "- candidate_answers: Candidate answers list field"
            )
        else:
            return "PseudoAnswerGenerator produces pseudo-answers through multi-round generation and selection."

    def run(self):
        # read input file : accept jsonl file only
        self.logger.info(f"Reading input file: {self.input_file}")
        # dataframe = pd.read_json(self.input_file,lines=True)
        dataframe = self.datastorage.read(self.input_file, "dataframe")
        input_data_number = dataframe.shape[0]
        # check if input_prompt_key are in the dataframe
        if self.input_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"read_key: {self.input_key} not found in the dataframe, please check the read_key: {key_list}")
        # check if output_text_key are in the dataframe
        if self.output_key_answer in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_answer} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        if self.output_key_solutions in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_solutions} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        if self.output_key_correct_solution_example in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_correct_solution_example} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        if self.output_key_answer_value in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_answer_value} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        # generate text
        user_prompts = dataframe[self.input_key].tolist()
        answer_dict = defaultdict(list)
        solution_dict = defaultdict(list)
        self.logger.info(f"Generating answers for {len(user_prompts)} questions")
        for i in range(self.max_times):
            self.logger.info(f"Generating: {i+1} times")
            solutions = self.model.generate_from_input(user_prompts)
            answers = [self.extractor.extract_answer(solution, None) for solution in solutions]
            for idx, answer in enumerate(answers):
                answer_dict[idx].append(answer)
                solution_dict[idx].append((answer, solutions[idx]))
        self.logger.info(f"Generating final answers")
        dataframe[self.output_key_answer] = dataframe.get(self.output_key_answer, None) 
        dataframe[self.output_key_solutions] = dataframe.get(self.output_key_solutions, None) 
        dataframe[self.output_key_correct_solution_example] = dataframe.get(self.output_key_correct_solution_example, None) 
        for key, value in answer_dict.items():
            count = Counter(value)
            final_answer = count.most_common(1)[0][0]
            dataframe.at[int(key),self.output_key_answer] = value
            dataframe.at[int(key),self.output_key_solutions] = final_answer
            correct_contents = [content for ans, content in solution_dict[key] if ans == final_answer]
            dataframe.at[int(key), self.output_key_solutions] = correct_contents
            correct_solution_example = correct_contents[0] if correct_contents else None
            dataframe.at[int(key), self.output_key_correct_solution_example] = correct_solution_example
            dataframe.at[int(key), self.output_key_answer_value] = final_answer
        # 过滤掉没有答案的行
        dataframe = dataframe[dataframe[self.output_key_answer_value].notna()]
        dataframe = dataframe[dataframe[self.output_key_correct_solution_example].notna()]
        self.logger.info(f"Data number {input_data_number} -> {dataframe.shape[0]}")
        self.datastorage.write(self.output_file, dataframe)
from typing import Dict, Union
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataflow.prompts.text2sql import QuestionRefinePrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class QuestionRefiner(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, num_threads: int = 5, max_retries: int = 3): 
        self.llm_serving = llm_serving       
        self.prompt = QuestionRefinePrompt()
        self.logger = get_logger()
        self.num_threads = num_threads
        self.max_retries = max_retries

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于对已有的自然语言问题进行润色改写。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题键\n"
                "- num_threads: 多线程并行数\n\n"
                "输出参数：\n"
                "- output_refined_question_key: 生成的润色后问题的key"
            )
        elif lang == "en":
            return (
                "This operator is used to refine and rewrite existing natural language questions.\n\n"
                "Input parameters:\n"
                "- input_question_key: Question key\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output parameters:\n"
                "- output_refined_question_key: The key for the generated refined question"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    def _generate_prompt(self, item: Dict) -> str:
        return self.prompt.question_refine_prompt(item['question'])
    
    def _parse_response(self, response: str, original_question: str) -> str:
        if not response:
            return original_question
            
        response_upper = response.upper()
        if "RESULT: NO" in response_upper:
            return original_question
            
        try:
            result_line = next(
                line for line in response.split('\n') 
                if line.upper().startswith("RESULT:")
            )
            return result_line.split("RESULT:", 1)[1].strip()
        except (StopIteration, IndexError):
            self.logger.warning(f"Unexpected response format: {response[:200]}...")
            return original_question

    def _process_item_with_retry(self, item: Dict, retry_count: int = 2) -> Dict:
        try:
            prompt = self._generate_prompt(item)
            response = self.llm_serving.generate_from_input([prompt])
            parsed_response = self._parse_response(response[0], item['question'])
            
            return {
                **item,
                self.output_refined_question_key: parsed_response
            }
        
        except Exception as e:
            if retry_count < self.max_retries:
                self.logger.warning(f"Retrying {item.get('id')} (attempt {retry_count + 1}): {str(e)}")
                return self._process_item_with_retry(item, retry_count + 1)

            return {
                **item,
                self.output_refined_question_key: item['question']
            }

    def run(self, storage: DataFlowStorage,
            input_question_key: str = "question",
            output_refined_question_key: str = "refined_question"
        ):
        self.input_question_key = input_question_key
        self.output_refined_question_key = output_refined_question_key

        self.logger.info("Starting QuestionRefiner...")
        raw_dataframe = storage.read("dataframe")
        items = raw_dataframe.to_dict('records')
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_item_with_retry, item): idx
                for idx, item in enumerate(tqdm(items, desc="Submitting tasks", unit="item"))
            }

            results = [None] * len(items)
            
            with tqdm(total=len(items), desc="Processing items", unit="item") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        self.logger.error(f"Fatal error for index={idx}: {e}")
                        results[idx] = {
                            **items[idx], 
                            self.output_refined_question_key: items[idx][self.input_question_key]
                        }
                    pbar.update(1)

        results = [r for r in results if r is not None]
        
        if len(results) != len(items):
            self.logger.warning(f"Results count mismatch: expected {len(items)}, got {len(results)}")
        
        output_file = storage.write(pd.DataFrame(results))
        self.logger.info(f"Refined questions saved to {output_file}")

        return [self.output_refined_question_key]

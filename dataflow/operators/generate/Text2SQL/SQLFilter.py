from tqdm import tqdm
import sqlite3
import sys
import re
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from func_timeout import func_timeout, FunctionTimedOut
from dataflow.prompts.text2sql import TextSQLConsistencyPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class SQLFilter(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, db_root_path: str, num_cpus: int = 20, meta_time_out: int = 120):
        self.llm_serving = llm_serving     
        self.prompt = TextSQLConsistencyPrompt()
        self.db_root_path = db_root_path
        self.num_cpus = num_cpus
        self.meta_time_out = meta_time_out
        self.logger = get_logger()
        
    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于筛选SQL执行异常和模型判定SQL与自然语言问题是否一致。\n\n"
                "输入参数：\n"
                "- input_sql_key: 输入数据中SQL语句的字段名\n"
                "- input_question_key: 输入数据中自然语言问题的字段名\n"
                "- input_dbid_key: 输入数据中数据库ID的字段名\n"
                "- db_root_path: 数据库文件的根目录路径\n"
                "- num_cpus: 并行线程数\n"
                "- meta_time_out: SQL执行超时时间"
            )
        elif lang == "en":
            return (
                "This operator filters SQL execution errors and checks the consistency between SQL and natural language questions using a model.\n\n"
                "Input parameters:\n"
                "- input_sql_key: Field name for SQL statements in input data (default: 'SQL')\n"
                "- input_question_key: Field name for questions in input data (default: 'question')\n"
                "- input_dbid_key: Field name for database IDs in input data (default: 'db_id')\n"
                "- db_root_path: Root directory path for database files (passed to constructor)\n"
                "- num_cpus: Number of parallel threads (passed to constructor, default 20)\n"
                "- meta_time_out: SQL execution timeout in seconds (passed to constructor, default 120)"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    @staticmethod
    def execute_sql(sql, db_path):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            return result
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def execute_model(ground_truth, db_place, idx, meta_time_out, logger):
        is_correct = True
        try:
            results = func_timeout(meta_time_out, SQLFilter.execute_sql,
                        args=(ground_truth, db_place))
            return {"idx": idx,"is_correct": is_correct, "results": results}
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
            sys.exit(0)
        except FunctionTimedOut:
            logger.info(f"timeout when execute idx {idx}")
            result = (f'timeout')
            is_correct = False
            return {"idx": idx,"is_correct": is_correct, "results": result}
        except Exception as e:
            logger.info(f"error: {e} when execute idx {idx}")
            result = (f'error:{e}')
            is_correct = False
            return {"idx": idx,"is_correct": is_correct, "results": result}
        
    def _parse_consistency_response(self, response):
        try:
            response_lower = response.lower() if response else ""
            
            conclusion = None
            reason = response
            
            if "conclusion:" in response_lower:
                conclusion_part = response_lower.split("conclusion:")[1].strip()
                
                if "no" in conclusion_part:
                    conclusion = False
                elif "yes" in conclusion_part:
                    conclusion = True
                else:
                    raise ValueError("Could not determine conclusion from response")
            else:
                raise ValueError("Response does not contain 'conclusion:'")
            
            return conclusion, reason
        except Exception as e:
            self.logger.warning(f"Failed to parse consistency response: {e}")
            return None, f"Parse error: {str(e)}"
        
    def run_sqls_parallel(self, datas, db_root_path, num_cpus, meta_time_out):
        exec_result = []
        pbar = tqdm(total=len(datas), desc="Executing SQLs")

        def wrap_task(ground_truth, db_place, idx, timeout):
            try:
                return SQLFilter.execute_model(ground_truth, db_place, idx, timeout, self.logger)
            except Exception as e:
                self.logger.error(f"Error executing SQL idx={idx}: {e}")
                return {"idx": idx, "is_correct": False, "results": f"error: {str(e)}"}

        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = []

            for i, data_pair in enumerate(datas):
                ground_truth = data_pair[self.input_sql_key]
                db_id = data_pair[self.input_dbid_key].replace('\n', '')
                db_id = re.sub(r'[^A-Za-z0-9_]', '', db_id)
                db_place = os.path.join(db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")

                future = executor.submit(wrap_task, ground_truth, db_place, i, meta_time_out)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    exec_result.append(result)
                except Exception as e:
                    self.logger.error(f"Error retrieving result from future: {e}")
                pbar.update()

        pbar.close()
        return sorted(exec_result, key=lambda x: x['idx'])
        
    def _reformat_prompt(self, dataframe):
        formatted_prompts = []
        for index, row in dataframe.iterrows():
            sql = row[self.input_sql_key]
            question = row[self.input_question_key]
            used_prompt = self.prompt.text_sql_consistency_prompt(question, sql)
            formatted_prompts.append(used_prompt.strip())
        return formatted_prompts
    
    def process_single_question(self, question):
        try:
            result = self.llm_serving.generate_from_input([question])
            return result[0] if result else ""
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return ""

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "SQL",
            input_dbid_key: str = "db_id",
            input_question_key: str = "question"
        ): 
        
        self.input_sql_key = input_sql_key
        self.input_dbid_key = input_dbid_key
        self.input_question_key = input_question_key

        dataframe = storage.read("dataframe")
        original_count = len(dataframe)

        datas = dataframe.to_dict('records')
        exec_results = self.run_sqls_parallel(datas, self.db_root_path, self.num_cpus, self.meta_time_out)
        
        sql_failed_count = 0
        sql_success_indices = []
        
        for exec_result in exec_results:
            idx = exec_result['idx']
            is_correct = exec_result.get('is_correct', False)
            
            if idx < len(dataframe) and is_correct:
                sql_success_indices.append(idx)
            else:
                sql_failed_count += 1
        
        self.logger.info(f"SQL execution results: {len(sql_success_indices)} passed, {sql_failed_count} failed")
        
        if not sql_success_indices:
            self.logger.warning("No SQL statements executed successfully. Returning empty dataset.")
            empty_df = dataframe.iloc[0:0].copy()
            output_file = storage.write(empty_df)
            return []
        
        self.logger.info("Step 2: Checking consistency between questions and SQL...")
        sql_passed_df = dataframe.loc[sql_success_indices].copy()
        formatted_prompts = self._reformat_prompt(sql_passed_df)
        
        try:
            responses = self.llm_serving.generate_from_input(formatted_prompts)
            
            if len(responses) != len(formatted_prompts):
                self.logger.warning(f"Expected {len(formatted_prompts)} responses but got {len(responses)}")
                while len(responses) < len(formatted_prompts):
                    responses.append("")
        except Exception as e:
            self.logger.error(f"Error in batch LLM processing: {e}")
            responses = [""] * len(formatted_prompts)
        
        consistency_failed_count = 0
        consistency_passed_count = 0
        final_valid_indices = []
        
        for i, (original_idx, response) in enumerate(zip(sql_success_indices, responses)):
            conclusion, reason = self._parse_consistency_response(response)
            
            if conclusion is True:
                consistency_passed_count += 1
                final_valid_indices.append(original_idx)
            else:
                consistency_failed_count += 1
        
        if final_valid_indices:
            filtered_dataframe = dataframe.loc[final_valid_indices].copy()
        else:
            self.logger.warning("No data passed all filters. Returning empty dataset.")
            filtered_dataframe = dataframe.iloc[0:0].copy()
        
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtered dataset saved to {output_file}")

        return []
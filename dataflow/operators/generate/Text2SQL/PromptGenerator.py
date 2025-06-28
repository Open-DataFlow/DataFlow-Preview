from typing import Dict, Union, Optional, Tuple    
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sqlite3
import os
import re
from tqdm import tqdm
from dataflow.prompts.text2sql import Text2SQLCotPrompt
from dataflow.prompts.text2sql import FinalPromptGeneration
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class PromptGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, db_root_path: str, num_threads: int = 5, timeout: int = 60):
        self.llm_serving = llm_serving
        self.prompt = FinalPromptGeneration()
        self.cot_output = Text2SQLCotPrompt()
        self.num_threads = num_threads
        self.timeout = timeout
        self.db_root_path = db_root_path
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于构建完整的提示词和思维链推理过程。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题键（如：question）\n"
                "- input_sql_key: SQL语句键（如：SQL）\n"
                "- input_schema_key: 数据库DDL信息键（如：ddl）\n"
                "- input_evidence_key: 输入中额外知识的键（如：evidence）\n"
                "- prompt_type: 提示词格式（如：omni-sql）\n"
                "- output_sft_prompt_key: SFT提示词输出键（如：sft_prompt）\n"
                "- output_rl_prompt_key: RL提示词输出键（如：rl_prompt）\n"
                "- output_cot_key: 思维链推理输出键（如：sft_output）\n"
                "- input_key: 输入数据主键（如：data）\n"
                "- input_dbid_key: 数据库ID键（如：db_id）\n"
                "- db_root_path: 数据库根目录（如：/mnt/public/data/.../dev_databases）\n"
                "- num_threads: 多线程并行数\n\n"
                "输出参数：\n"
                "- output_sft_prompt_key: SFT提示词\n"
                "- output_rl_prompt_key: RL提示词\n"
                "- output_cot_key: 思维链推理输出"
            )
        elif lang == "en":
            return (
                "This operator is used to construct complete prompts and chain-of-thought reasoning processes.\n\n"
                "Input parameters:\n"
                "- input_question_key: Key for the question (e.g., 'question')\n"
                "- input_sql_key: Key for the SQL statement (e.g., 'SQL')\n"
                "- input_schema_key: Key for the database DDL information (e.g., 'ddl')\n"
                "- input_evidence_key: Key for additional knowledge in the input (e.g., 'evidence')\n"
                "- prompt_type: Prompt format (e.g., 'omni-sql')\n"
                "- output_sft_prompt_key: Output key for SFT prompt (e.g., 'sft_prompt')\n"
                "- output_rl_prompt_key: Output key for RL prompt (e.g., 'rl_prompt')\n"
                "- output_cot_key: Output key for chain-of-thought reasoning (e.g., 'sft_output')\n"
                "- input_key: Main key for input data (e.g., 'data')\n"
                "- input_dbid_key: Key for database ID (e.g., 'db_id')\n"
                "- db_root_path: Root path of the databases (e.g., '/mnt/public/data/.../dev_databases')\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output parameters:\n"
                "- output_sft_prompt_key: SFT prompt\n"
                "- output_rl_prompt_key: RL prompt\n"
                "- output_cot_key: Chain-of-thought reasoning output"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    def generate_prompt(self, item: Dict, prompt_type: str) -> str:
        generated_prompt = None
        if prompt_type == 'dail-sql':
            generated_prompt = self.prompt.dial_sql_cot_prompt(
                    question=item.get(self.input_question_key),
                    schema=item.get(self.input_schema_key)
            )
        elif prompt_type == 'omni-sql':
            generated_prompt = self.prompt.omni_sql_cot_prompt(
                    question=item.get(self.input_question_key),
                    schema=item.get(self.input_schema_key)
            )
        return generated_prompt
    
    def generate_cot_synthesis_prompts(self, item: Dict, is_backup=False) -> str:
        if not is_backup:
            cot_synthesis_prompt = self.cot_output.text2sql_cot_prompt(
                item.get(self.input_schema_key),
                item.get(self.input_question_key),
                item.get(self.input_sql_key)
            )
        else:
            cot_synthesis_prompt = self.cot_output.text2sql_cot_prompt_backup(
                item.get(self.input_schema_key),
                item.get(self.input_question_key),
                item.get(self.input_sql_key)
            )
    
        return cot_synthesis_prompt
    
    def execute_sql(self, sql, db_path, timeout=10):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout = 5000")
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            self.logger.error(f"SQL执行错误: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def extract_sql(self, response):
        pattern = r"```sql\s*(.*?)\s*```"
        
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            return ""
    
    def _parse_response(self, response: str, gold_sql: str, db_path) -> Tuple[Optional[str], bool]:
        generated_sql = self.extract_sql(response)
        if not generated_sql:
            return None, False
        
        try:
            gen_result = self.execute_sql(generated_sql, db_path)
            gold_result = self.execute_sql(gold_sql, db_path)
            
            if gen_result is None or gold_result is None:
                return generated_sql, False
                
            return generated_sql, gen_result == gold_result 
        except Exception as e:
            self.logger.warning(f"SQL执行失败: {e}")
            return generated_sql, False

    def _parse_backup_response(self, response: str) -> Tuple[Optional[str], bool]:
        response = response.strip()
        if not response:
            return None, False

        lower_response = response.lower()
        keywords = ["let"] 
        
        for keyword in keywords:
            idx = lower_response.find(keyword)
            if idx != -1:
                return response[idx:], True
        
        return None, False

    def _process_item_with_retry(self, item: Dict, retry_count: int = 0, max_retries: int = 3) -> str:
        db_id = item.get(self.input_dbid_key)
        gold_sql = item.get(self.input_sql_key)
        db_path = os.path.join(self.db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")
        prompt = self.generate_cot_synthesis_prompts(item, False)
        
        while retry_count <= max_retries:
            try:
                response = self.llm_serving.generate_from_input([prompt])
                parsed_response, flag = self._parse_response(response[0], gold_sql, db_path)
                
                if flag:
                    return parsed_response if parsed_response else ""
                
                retry_count += 1
            except Exception as e:
                self.logger.warning(f"Attempt {retry_count} failed: {e}")
                retry_count += 1

        try:
            backup_prompt = self.generate_cot_synthesis_prompts(item, True)
            backup_response = self.llm_serving.generate_from_input([backup_prompt])
            parsed_backup_response, success = self._parse_backup_response(backup_response[0])
            return parsed_backup_response if success and parsed_backup_response else ""
        except Exception as e:
            self.logger.error(f"Backup processing failed: {e}")
            return ""

    
    def _process_item(self, item: Dict) -> Dict: 
        sft_prompt = self.generate_prompt(item, prompt_type="omni-sql")
        rl_prompt = self.generate_prompt(item, prompt_type="dail-sql")
        cot_output = self._process_item_with_retry(item)

        return {
            **item,
            self.output_sft_prompt_key: sft_prompt if sft_prompt else '',
            self.output_rl_prompt_key: rl_prompt if rl_prompt else '',
            self.output_cot_key: cot_output if cot_output else ''
        }

    def run(self, storage: DataFlowStorage, 
            input_sql_key: str = "SQL",
            input_question_key: str = "question",
            input_dbid_key: str = "db_id",
            input_schema_key: str = "ddl",
            output_sft_prompt_key: str = "sft_prompt",
            output_rl_prompt_key: str = "rl_prompt",
            output_cot_key: str = "sft_output"
        ):
        self.input_question_key = input_question_key
        self.input_sql_key = input_sql_key
        self.input_schema_key = input_schema_key
        self.input_dbid_key = input_dbid_key
        self.output_sft_prompt_key = output_sft_prompt_key
        self.output_rl_prompt_key = output_rl_prompt_key
        self.output_cot_key = output_cot_key
        
        self.logger.info("Starting prompt generation...")
        raw_dataframe = storage.read("dataframe")
        items = raw_dataframe.to_dict('records')
            
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_item, item): idx
                for idx, item in enumerate(tqdm(items, desc="Submitting tasks", unit="item"))
            }

            results = [None] * len(items)
            
            with tqdm(total=len(futures), desc="Processing", unit="item") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing index={idx}: {e}")
                        results[idx] = {
                            **items[idx], 
                            self.output_sft_prompt_key: "", 
                            self.output_rl_prompt_key: "",
                            self.output_cot_key: ""
                        }
                    
                    pbar.update(1)

        final_results = [r for r in results if r is not None]
        
        if len(final_results) != len(items):
            self.logger.warning(f"Results count mismatch: expected {len(items)}, got {len(final_results)}")
        
        output_file = storage.write(pd.DataFrame(final_results))
        self.logger.info(f"Prompt generation completed, saved to {output_file}")

        return [self.output_sft_prompt_key, self.output_rl_prompt_key, self.output_cot_key]

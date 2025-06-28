from typing import Dict, Union, List
from tqdm import tqdm
import pandas as pd
from dataflow.prompts.text2sql import ExtraKnowledgePrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class ExtraKnowledgeGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, exist_knowledge: bool = False, max_retries: int =2, batch_size: int = 50):        
        self.num_threads = 20
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.llm_serving = llm_serving
        self.exist_knowledge = exist_knowledge
        self.logger = get_logger()
        self.prompt = ExtraKnowledgePrompt()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于生成基于构建SQL生成所需的额外知识\n\n"
                "输入参数：\n"
                "- input_question_key: 自然语言问题的key\n"
                "- input_sql_key: SQL的key\n"
                "- input_schema_key: Schema提取那一步中提取出的完整Schema的key\n"
                "- exist_knowledge: 当前数据集是否存在 knowledge，True or False\n"
                "- max_retries: 提取额外知识的最多尝试时间，若某次提取失败，则会尝试再次生成并提取\n"
                "- num_threads: 多线程并行数\n\n"
                "输出参数：\n"
                "- output_knowledge_key: 生成的额外知识的key"
            )
        elif lang == "en":
            return (
                "This operator is used to generate additional knowledge required for SQL generation based on constructed SQL.\n\n"
                "Input parameters:\n"
                "- input_question_key: Key for natural language questions\n"
                "- input_sql_key: Key for SQL statements\n"
                "- input_schema_key: Key for the complete schema extracted in the schema extraction step\n"
                "- exist_knowledge: Whether the current dataset already contains knowledge, True or False\n"
                "- max_retries: Maximum retry attempts for extracting additional knowledge; if an extraction fails, it will attempt to generate and extract again\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output parameters:\n"
                "- output_knowledge_key: Key for the generated additional knowledge"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    def _generate_prompt(self, item: Dict) -> str:
        return self.prompt.extra_knowledge_prompt(item[self.input_question_key],
                                                  item[self.input_sql_key],
                                                  item[self.input_schema_key])
    
    def _parse_response(self, response: str) -> str:
        response = response.strip()
        upper_response = response.strip().upper()
        if not upper_response:
            return None
        
        if "RESULT: NO" in upper_response:
            return None
        
        try:
            result_line = next(
                line for line in response.split('\n') 
                if line.strip().startswith("RESULT:")
            )
            knowledge = result_line.split("RESULT:", 1)[1].strip()
            return knowledge if knowledge else None
        except (StopIteration, IndexError):
            self.logger.warning(f"Failed to parse response: {response[:200]}...")
            return None
    
    def _process_batch(self, batch_items: List[Dict], retry_count: int = 0) -> List[Dict]:
        try:
            prompts = [self._generate_prompt(item) for item in batch_items]
            responses = self.llm_serving.generate_from_input(prompts)
            
            results = []
            for item, response in zip(batch_items, responses):
                parsed_response = self._parse_response(response)
                results.append({
                    **item,
                    self.output_knowledge_key: parsed_response
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            
            if retry_count < self.max_retries:
                self.logger.warning(f"Retrying batch (attempt {retry_count + 1})")
                return self._process_batch(batch_items, retry_count + 1)
            else:
                self.logger.warning("Batch processing failed, falling back to individual processing")
                results = []
                for item in batch_items:
                    try:
                        prompt = self._generate_prompt(item)
                        response = self.llm_serving.generate_from_input([prompt])
                        parsed_response = self._parse_response(response[0])
                        results.append({
                            **item,
                            self.output_knowledge_key: parsed_response
                        })
                    except Exception as e:
                        self.logger.error(f"Individual processing failed for id={item.get('id')}: {e}")
                        results.append({
                            **item,
                            self.output_knowledge_key: ''
                        })
                return results

    def run(self, storage: DataFlowStorage,
            input_question_key: str = "question",
            input_sql_key: str = "SQL",
            input_schema_key: str = "ddl",
            output_knowledge_key: str = "evidence"
        ):
        self.input_question_key = input_question_key
        self.input_sql_key = input_sql_key
        self.input_schema_key = input_schema_key
        self.output_knowledge_key = output_knowledge_key

        self.logger.info("Starting ExtraKnowledgeGenerator...")
        raw_dataframe = storage.read("dataframe")
        items = raw_dataframe.to_dict(orient='records')
        
        if items:
            existing_count = sum(1 for item in items if self.output_knowledge_key in item and item[self.output_knowledge_key])
            if existing_count == len(items):
                self.logger.info("Extra knowledge already exists for all items, skipping generation.")
                output_file = storage.write(pd.DataFrame(items))
                return []
            elif existing_count > 0:
                self.logger.info(f"Extra knowledge exists for {existing_count}/{len(items)} items, processing remaining items.")
        
        indexed_items = [(idx, item) for idx, item in enumerate(items)]
        batches = [indexed_items[i:i + self.batch_size] for i in range(0, len(indexed_items), self.batch_size)]
        self.logger.info(f"Processing {len(items)} items in {len(batches)} batches of size {self.batch_size}")
        
        all_results = [None] * len(items)
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} items")
            batch_to_process = []
            skip_indices = []
            
            for idx, item in batch:
                if self.output_knowledge_key in item and item[self.output_knowledge_key]:
                    all_results[idx] = item
                    skip_indices.append(idx)
                else:
                    batch_to_process.append((idx, item))
            
            if batch_to_process:
                items_only = [item for _, item in batch_to_process]
                batch_results = self._process_batch(items_only)
                
                for (original_idx, _), result in zip(batch_to_process, batch_results):
                    all_results[original_idx] = result
            
            if skip_indices:
                self.logger.info(f"Skipped {len(skip_indices)} items that already have knowledge in batch {batch_idx + 1}")
        
        final_results = []
        for idx, result in enumerate(all_results):
            if result is None:
                self.logger.warning(f"Missing result for index {idx}, using original item")
                final_results.append({
                    **items[idx],
                    self.output_knowledge_key: ''
                })
            else:
                final_results.append(result)
        
        output_file = storage.write(pd.DataFrame(final_results))
        self.logger.info(f"Knowledge generation completed, saved to {output_file}")

        return [self.output_knowledge_key]
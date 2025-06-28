import pandas as pd
import os
import sqlite3
from tqdm import tqdm
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class DatabaseSchemaExtractor(OperatorABC):
    def __init__(self, table_info_file: str, db_root_path: str, num_threads: int = 5):
        self.table_schema_path = table_info_file
        self.database_base_path = db_root_path
        self.num_threads = num_threads
        self._schema_cache = {}
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于提取出数据库模式和数据库定义语言，并将其格式化输出。\n\n"
                "输入参数：\n"
                "- table_schema_path: 输入的table schema文件路径\n"
                "- table_schema_file_db_key: 输入的table schema文件中db的key\n"
                "- input_db_key: 输入文件中每条数据对应的db的key\n"
                "- database_base_path: 数据库路径\n"
                "- input_question_key: Question key\n"
                "- input_sql_key: SQL key\n"
                "- selected_schema_key: 上一步Schema Linking中提取出的shcema的key\n"
                "- num_threads: 多线程并行数\n\n"
                "输出参数：\n"
                "- output_raw_schema_key: 原始的shcema中，没有文本格式，用json格式保存\n"
                "- output_ddl_key: 所在数据库的ddl\n"
                "- output_whole_format_schema_key: 完整的shcema构成的格式化信息，包括schema、example等\n"
                "- output_selected_format_schema_key: 由Schema Linking筛选出来的schema构成的格式化数据库信息"
            )
        elif lang == "en":
            return (
                "This operator extracts the database schema and database definition language, and formats the output.\n\n"
                "Input Parameters:\n"
                "- table_schema_path: Input table schema file path\n"
                "- table_schema_file_db_key: Key in the input table schema file for db\n"
                "- input_db_key: Key in the input file for each data's db\n"
                "- database_base_path: Database path\n"
                "- input_question_key: Question key\n"
                "- input_sql_key: SQL key\n"
                "- selected_schema_key: Key of the schema extracted by Schema Linking\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output Parameters:\n"
                "- output_raw_schema_key: Raw schema without text format, saved in JSON format\n"
                "- output_ddl_key: DDL of the database\n"
                "- output_whole_format_schema_key: Formatted information of the complete schema, including schema and examples\n"
                "- output_selected_format_schema_key: Formatted database information of the selected schema by Schema Linking"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    def collect_schema(self, db_id: str) -> Dict:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        table_schema = pd.read_json(self.table_schema_path, lines=True).to_dict(orient='records')
        
        for schema in table_schema:
            if schema[self.table_schema_file_db_key] == db_id:
                self._schema_cache[db_id] = schema
                return schema
        return None

    def extract_schema(self, db_info: Dict, db_conn: sqlite3.Connection) -> Dict:
        schema = {
            'tables': {},
            'foreign_keys': [],
            'primary_keys': []
        }
        
        table_names = db_info["table_names_original"]
        cursor = db_conn.cursor()
        
        try:
            for i in range(len(db_info["column_names_original"])):
                if db_info["column_names_original"][i][0] == -1:
                    continue
                    
                table_idx = db_info["column_names_original"][i][0]
                table_name = table_names[table_idx]
                col_name = db_info["column_names_original"][i][1]
                col_type = db_info["column_types"][i]
                
                if table_name not in schema['tables']:
                    schema['tables'][table_name] = {
                        'columns': {},
                        'primary_keys': []
                    }
                
                schema['tables'][table_name]['columns'][col_name] = {
                    'type': col_type,
                    'examples': []
                }
                
                try:
                    sql_query = f'SELECT "{col_name}" FROM "{table_name}" LIMIT 2'
                    cursor.execute(sql_query)
                    examples = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
                    schema['tables'][table_name]['columns'][col_name]['examples'] = examples
                except sqlite3.Error as e:
                    self.logger.warning(f"Unable to access examples for {table_name}.{col_name}: {e}")
            
            for pk in db_info["primary_keys"]:
                if isinstance(pk, list):  
                    table_idxs = {db_info["column_names_original"][col_idx][0] for col_idx in pk}
                    if len(table_idxs) != 1:
                        continue 
                    
                    table_idx = table_idxs.pop()
                    if table_idx == -1:
                        continue
                        
                    table_name = table_names[table_idx]
                    col_names = [
                        db_info["column_names_original"][col_idx][1]
                        for col_idx in pk
                    ]
                    
                    if table_name in schema['tables']:
                        schema['tables'][table_name]['primary_keys'].extend(col_names)
                        schema['primary_keys'].append({
                            'table': table_name,
                            'columns': col_names
                        })
                else:
                    table_idx = db_info["column_names_original"][pk][0]
                    if table_idx == -1:
                        continue
                        
                    table_name = table_names[table_idx]
                    col_name = db_info["column_names_original"][pk][1]
                    
                    if table_name in schema['tables']:
                        schema['tables'][table_name]['primary_keys'].append(col_name)
                        schema['primary_keys'].append({
                            'table': table_name,
                            'column': col_name
                        })
            
            for fk in db_info["foreign_keys"]:
                src_col_idx, ref_col_idx = fk
                
                src_table_idx = db_info["column_names_original"][src_col_idx][0]
                ref_table_idx = db_info["column_names_original"][ref_col_idx][0]
                
                if src_table_idx == -1 or ref_table_idx == -1:
                    continue
                    
                src_table = table_names[src_table_idx]
                src_col = db_info["column_names_original"][src_col_idx][1]
                ref_table = table_names[ref_table_idx]
                ref_col = db_info["column_names_original"][ref_col_idx][1]
                
                schema['foreign_keys'].append({
                    'source_table': src_table,
                    'source_column': src_col,
                    'referenced_table': ref_table,
                    'referenced_column': ref_col
                })
                
        finally:
            cursor.close()
            
        return schema

    def generate_ddl_from_schema(self, schema: Dict) -> str:
        ddl_statements = []
        
        for table_name, table_info in schema['tables'].items():
            columns_ddl = []
            
            for col_name, col_info in table_info['columns'].items():
                sql_type = {
                    "number": "INTEGER",
                    "text": "TEXT",
                    "date": "DATE",
                    "time": "TIME",
                    "datetime": "DATETIME"
                }.get(col_info['type'].lower(), "TEXT")
                
                columns_ddl.append(f"    {col_name} {sql_type}")
            
            if table_info['primary_keys']:
                pk_columns = ", ".join(table_info['primary_keys'])
                columns_ddl.append(f"    PRIMARY KEY ({pk_columns})")
            
            for fk in schema['foreign_keys']:
                if fk['source_table'] == table_name:
                    columns_ddl.append(
                        f"    FOREIGN KEY ({fk['source_column']}) "
                        f"REFERENCES {fk['referenced_table']}({fk['referenced_column']})"
                    )
            
            create_table_sql = (
                f"CREATE TABLE {table_name} (\n" +
                ",\n".join(columns_ddl) +
                "\n);"
            )
            ddl_statements.append(create_table_sql)
        
        return "\n\n".join(ddl_statements)

    def generate_formatted_schema(self, schema: Dict) -> str:
        formatted = []
        
        for table_name, table_info in schema['tables'].items():
            formatted.append(f"## Table: {table_name}")
            
            if table_info['primary_keys']:
                formatted.append(f"Primary Key: {', '.join(table_info['primary_keys'])}")
            
            formatted.append("Column Information:")
            for col_name, col_info in table_info['columns'].items():
                examples = ", ".join(col_info['examples']) if col_info['examples'] else ""
                formatted.append(
                    f"- {col_name} ({col_info['type']}) "
                    f"Example: {examples}"
                )
            
            table_fks = [
                fk for fk in schema['foreign_keys'] 
                if fk['source_table'] == table_name
            ]
            if table_fks:
                formatted.append("Foreign Key:")
                for fk in table_fks:
                    formatted.append(
                        f"- {fk['source_column']} → "
                        f"{fk['referenced_table']}.{fk['referenced_column']}"
                    )
            
            formatted.append("") 
        
        return "\n".join(formatted)
    
    def generate_selected_format_schema(self, selected_schema: Dict) -> str:
        selected_format_schema = ""
        for table in selected_schema:
            table_name = table["table_name"]
            columns = ",".join(table["columns"])
            selected_format_schema += f"Table {table_name}, columns = [{columns}]\n"

        return selected_format_schema
    
    def _process_item(self, item: Dict) -> Dict:
        try:
            db_id = item[self.input_db_key]
            db_info = self.collect_schema(db_id)
            
            db_path = os.path.join(self.database_base_path, db_id, f"{db_id}.sqlite")
            with sqlite3.connect(db_path) as db_conn:
                schema = self.extract_schema(db_info, db_conn)
                return {
                    **item, 
                    self.output_raw_schema_key: schema,
                    self.output_ddl_key: self.generate_ddl_from_schema(schema),
                    self.output_whole_format_schema_key: self.generate_formatted_schema(schema)
                }
        except Exception as e:
            self.logger.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
            return {
                **item,
                self.output_raw_schema_key: "",
                self.output_ddl_key: "",
                self.output_whole_format_schema_key: ""
            }

    def run(self, storage: DataFlowStorage,
            input_db_key: str = "db_id", 
            table_schema_file_db_key: str = "db_id",
            output_raw_schema_key: str = "whole_schema",
            output_ddl_key: str = "ddl",
            output_whole_format_schema_key: str ="whole_format_schema"
        ):
        self.input_db_key = input_db_key
        self.output_raw_schema_key = output_raw_schema_key
        self.output_ddl_key = output_ddl_key
        self.output_whole_format_schema_key = output_whole_format_schema_key
        self.table_schema_file_db_key = table_schema_file_db_key

        raw_dataframe = storage.read("dataframe")
        items = raw_dataframe.to_dict(orient='records')

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_item, item): idx 
                for idx, item in enumerate(tqdm(items, desc="Submitting tasks", unit="item"))
            }
            
            results = [None] * len(items)
            
            with tqdm(total=len(items), desc="Processing") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        self.logger.error(f"Error processing index={idx}: {e}")
                        error_result = items[idx].copy()
                        error_result['_error'] = str(e)
                        results[idx] = error_result
                        
                    pbar.update(1)
        
        results = [r for r in results if r is not None]
        
        output_file = storage.write(pd.DataFrame(results))
        self.logger.info(f"Extracted answers saved to {output_file}")

        return [self.output_raw_schema_key, self.output_ddl_key, self.output_whole_format_schema_key]
from tqdm import tqdm
import json
import re
from sqlglot.optimizer.qualify import qualify
from sqlglot import parse_one, exp
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


SQLITE_RESERVED_KEYWORDS = {
    "abort", "action", "add", "after", "all", "alter", "analyze", "and", "as", "asc", "attach", "autoincrement",
    "before", "begin", "between", "by", "cascade", "case", "cast", "check", "collate", "column", "commit", "conflict",
    "constraint", "create", "cross", "current_date", "current_time", "current_timestamp", "database", "default",
    "deferrable", "deferred", "delete", "desc", "detach", "distinct", "drop", "each", "else", "end", "escape", "except",
    "exclusive", "exists", "explain", "fail", "for", "foreign", "from", "full", "glob", "group", "having", "if",
    "ignore", "immediate", "in", "index", "indexed", "initially", "inner", "insert", "instead", "intersect", "into",
    "is", "isnull", "join", "key", "left", "like", "limit", "natural", "no", "not", "notnull", "null", "of",
    "offset", "on", "or", "order", "outer", "plan", "pragma", "primary", "query", "raise", "recursive", "references",
    "regexp", "reindex", "release", "rename", "replace", "restrict", "right", "rollback", "row", "savepoint", "select",
    "set", "table", "temp", "temporary", "then", "to", "trigger", "union", "unique", "update", "using",
    "vacuum", "values", "view", "virtual", "when", "where", "with", "without"
}


@OPERATOR_REGISTRY.register()
class SchemaLinking(OperatorABC):
    def __init__(self, table_info_file: str):
        self.table_info_file = table_info_file
        self.logger = get_logger()
        self.schema_cache = {}  

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于通过解析SQL语句提取使用的数据库Schema。\n\n"
                "输入参数：\n"
                "- table_info_file：tables.jsonl文件路径，包含数据库Schema信息\n"
                "- input_sql_key：SQL语句键\n"
                "- input_dbid_key：db_id key，数据库名\n\n"
                "输出参数：\n"
                "- output_used_schema_key：SQL中实际使用的表和列信息，格式为字典，键为表名，值为列名列表"
            )
        elif lang == "en":
            return (
                "This operator extracts used database schema by parsing SQL statements.\n\n"
                "Input parameters:\n"
                "- table_info_file: Path to tables.jsonl file containing database schema information\n"
                "- input_sql_key: SQL statement key\n"
                "- input_dbid_key: db_id key, database name\n\n"
                "Output parameters:\n"
                "- output_used_schema_key: Actually used tables and columns in SQL, formatted as dict with table names as keys and column lists as values"
            )
        else:
            return "Schema linking operator for Text2SQL tasks using sqlglot parsing."

    def load_schema_info(self):
        if self.schema_cache:
            return self.schema_cache
            
        try:
            with open(self.table_info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        schema_info = json.loads(line.strip())
                        db_id = schema_info['db_id']
                        
                        schema = {}
                        table_names_original = schema_info['table_names_original']
                        column_names_original = schema_info['column_names_original']
                        
                        for table_idx, table_name in enumerate(table_names_original):
                            schema[table_name.lower()] = []
                            
                        for col_info in column_names_original:
                            table_idx, col_name = col_info
                            if table_idx >= 0: 
                                table_name = table_names_original[table_idx].lower()
                                schema[table_name].append(col_name.lower())
                        
                        self.schema_cache[db_id] = schema
                        
        except Exception as e:
            self.logger.error(f"Error loading schema info from {self.table_info_file}: {e}")
            
        return self.schema_cache

    def get_schema_for_db(self, db_id):
        schema_cache = self.load_schema_info()
        return schema_cache.get(db_id, {})

    def normalize_sql_column_references(self, sql: str, schema: dict, alias_map: dict) -> str:
        col_to_table = {}
        all_tables = []
        
        for table, cols in schema.items():
            all_tables.append(table)
            for col in cols:
                if col not in col_to_table:
                    col_to_table[col] = []
                col_to_table[col].append(table)

        col_fix_map = {}
        for col, tables in col_to_table.items():
            if len(tables) == 1:
                table = tables[0]
                alias = None
                for a, t in alias_map.items():
                    if t == table:
                        alias = a
                        break
                if alias:
                    col_fix_map[col] = f'"{alias}"."{col}"'
                    
        alias_pattern1 = re.compile(r'\bAS\s+"?([a-zA-Z_][\w]*)"?', re.IGNORECASE)
        alias_names = set(m.group(1) for m in alias_pattern1.finditer(sql))
        
        alias_pattern2 = re.compile(r'\bAS\s+(?:"(?P<dq>[^"]+)"|`(?P<bq>[^`]+)`)', re.IGNORECASE)
        for m in alias_pattern2.finditer(sql):
            alias = m.group('dq') or m.group('bq')
            alias_names.add(alias)

        def replace_col(m):
            col = m.group(0).strip('"')
            bef = m.string[max(0, m.start()-10):m.start()]
            
            if ('.' in bef or col in all_tables or col in alias_names or 
                col in SQLITE_RESERVED_KEYWORDS):
                return m.group(0) 
                
            if ((m.group(0).startswith('"') and not m.group(0).endswith('"')) or
                (not m.group(0).startswith('"') and m.group(0).endswith('"'))):
                return m.group(0) 
                
            return col_fix_map.get(col, m.group(0))

        if col_fix_map:
            pattern = re.compile(
                r'(?<![\w.])("?(' + '|'.join(re.escape(c) for c in col_fix_map.keys()) + r')"?)(?![\w])'
            )
            sql = pattern.sub(replace_col, sql)

        return sql

    def extract_alias_table_map(self, ast, alias_map):
        for table in ast.find_all(exp.Table):
            table_name = table.name.lower() if table.name else ""
            if table.alias:
                alias_map[table.alias.lower()] = table_name
            else:
                alias_map[table_name] = table_name

        for subquery in ast.find_all(exp.Subquery):
            if subquery.alias:
                inner_table = subquery.this.find(exp.Table)
                source_name = inner_table.name.lower() if inner_table and inner_table.name else "subquery"
                alias_map[subquery.alias.lower()] = source_name

    def get_cols(self, ast, alias_map, schema, used_columns):
        columns = list(ast.find_all(exp.Column))
        col_to_table = {}
        
        for table, cols in schema.items():
            for col in cols:
                if col not in col_to_table:
                    col_to_table[col] = []
                col_to_table[col].append(table)
        
        stars = list(ast.find_all(exp.Star))
        for star in stars:
            for table_name in schema.keys():
                if table_name in alias_map.values():
                    for col in schema[table_name]:
                        used_columns.add((table_name, col))
                
        for col in columns:
            col_name = col.name.lower() if col.name else ""
            table_ref = col.table.lower() if col.table else ""
            
            if table_ref == '':
                if col_name in col_to_table:
                    for table in col_to_table[col_name]:
                        if table in alias_map.values():
                            used_columns.add((table, col_name))
                            break  
            else:
                if table_ref in alias_map and col_name in col_to_table:
                    actual_table = alias_map[table_ref]
                    if actual_table in schema and col_name in schema[actual_table]:
                        used_columns.add((actual_table, col_name))

    def convert_to_grouped_schema(self, used_columns):
        grouped_schema = {}
        for table, column in used_columns:
            if table not in grouped_schema:
                grouped_schema[table] = []
            if column not in grouped_schema[table]:
                grouped_schema[table].append(column)
        
        for table in grouped_schema:
            grouped_schema[table].sort()
            
        return grouped_schema

    def extract_used_schema_from_sql(self, sql, db_id):
        schema = self.get_schema_for_db(db_id)
        if not schema:
            self.logger.warning(f"No schema found for database {db_id}")
            return {}
            
        used_columns = set()
        
        try:
            sql = sql.replace('`', '"')
            ast = parse_one(sql.lower())
            alias_map = {}
            self.extract_alias_table_map(ast, alias_map)

            sql = self.normalize_sql_column_references(sql.lower(), schema, alias_map)
            
            ast = parse_one(sql.lower())
            try:
                qualify(ast)
            except:
                self.logger.debug("qualify error. Skip qualify.")
                
            alias_map = {}
            self.extract_alias_table_map(ast, alias_map)
            self.get_cols(ast, alias_map, schema, used_columns)
            
        except Exception as e:
            self.logger.error(f"Error parsing SQL for {db_id}: {sql[:100]}... Error: {e}")
            
        return self.convert_to_grouped_schema(used_columns)

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "SQL",
            input_dbid_key: str = "db_id",
            output_used_schema_key: str = "used_schema"
        ):
        
        self.load_schema_info()
        
        dataframe = storage.read("dataframe")
        
        required_cols = [input_sql_key, input_dbid_key]
        missing_cols = [col for col in required_cols if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        used_schemas = []
        failed_count = 0
        
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Extracting schema from SQL"):
            sql = row[input_sql_key]
            db_id = row[input_dbid_key]
            
            db_id_clean = re.sub(r'[^A-Za-z0-9_]', '', str(db_id).replace('\n', ''))
            
            used_schema = self.extract_used_schema_from_sql(sql, db_id_clean)
            if not used_schema:
                failed_count += 1
            used_schemas.append(used_schema)
        
        dataframe[output_used_schema_key] = used_schemas

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_used_schema_key]

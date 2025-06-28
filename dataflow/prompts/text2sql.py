'''
A collection of prompts for the text2sql operator.
'''
class TextSQLConsistencyPrompt:
    def __init__(self):
        pass

    def text_sql_consistency_prompt(self, question, sql):
        prompt = f"""
            ## SQL Consistency Verification Task
            
            **Objective**: Given the question and SQL query, determine if the SQL query correctly implements the requirements specified in the natural language Question.
            
            **Evaluation Criteria**:
            1. The SQL should reflect key elements from the Question:
            2. You can refer to the content in evidence to determine if the SQL meets the requirements of the question
            3. Since you are not given the database schema, you can only analyze the SQL query and its relation to the Question and evidence.
            4. Do not judge as inconsistent just because of the database schema
            
            **Input**:
            Question: {question}
            SQL: {sql}
            
            **Required Output Format**:
            Analysis: <Brief technical analysis of the alignment between Question and SQL>
            Conclusion: <"YES" if consistent or uncertain, "NO" if definitely inconsistent> (No other text)
            
            **Example**:
            Analysis: The SQL query correctly implements the requirements of the Question, (may be more).
            Conclusion: <YES>
            
            **Important Notes**:
            - Respond ONLY with the specified format above
            - "YES" should be used when SQL implements Question OR when you're uncertain
            - "NO" should be used when SQL contradicts the Question
            - Be strict with logical requirements but lenient with syntax variations
            """
        return prompt
    
class QuestionRefinePrompt:
    def __init__(self):
        pass

    def question_refine_prompt(self, question):
        """Refine the question"""
        prompt = (
            "Analyze the following question and determine if it needs clarification:\n"
            f"ORIGINAL QUESTION: {question}\n"
            "Instructions:\n"
            "1. If the question is already perfectly clear, output: 'NO'\n"
            "2. If clarification would help, rewrite it to be more precise while:\n"
            "   - Preserving all original meaning\n"
            "   - Not adding/removing any factual content\n"
            "   - Only improving clarity of expression\n\n"
            "Format your response exactly as:\n"
            "```\n"
            "ANALYSIS: <brief explanation of why rewrite is/isn't needed>\n"
            "RESULT: <either 'NO' or the rewritten question>\n"
            "```"
        )
        
        return prompt
    
class ExtraKnowledgePrompt:
    def __init__(self):
        pass

    def extra_knowledge_prompt(self, question, sql, schema):
        prompt = (
            "Analyze whether answering this database question requires additional knowledge beyond the provided SQL and schema.\n"
            f"QUESTION: {question}\n"
            f"SQL QUERY: {sql}\n"
            f"TABLE SCHEMA:\n{schema}\n\n"
            "Consider:\n"
            "1. Are there domain terms not explained in the schema?\n"
            "2. Does the query rely on implicit business rules?\n"
            "3. Is special knowledge needed to interpret results?\n\n"
            "Respond ONLY in this exact format:\n"
            "RESULT: <knowledge> OR RESULT: NO\n"
            "Where <knowledge> is a concise explanation of required additional knowledge.\n"
            "If no extra knowledge is needed, respond with exactly 'RESULT: NO'."
        )
        return prompt


class FinalPromptGeneration:
    def __init__(self):
        pass

    def dial_sql_cot_prompt(self, question, schema):
        prompt = (
            "/* Given the following database schema: */\n"
            f"{schema}\n\n"
            f"/* Answer the following: {question} */\n"
            "Let's think step by step ",
        )

        return prompt
    
    def dial_sql_non_cot_prompt(self, question, schema):
        prompt = (
            "/* Given the following database schema: */\n"
            f"{schema}\n\n"
            f"/* Answer the following: {question} */\n"
            "SELECT ",
        )

        return prompt
    
    def omni_sql_cot_prompt(self, question, schema):
        prompt = (
            "Task Overview:\n"
            "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.\n\n"
            "Database Engine:\n"
            "SQLite\n\n"
            "Database Schema:\n"
            f"{schema}\n"
            "This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Instructions:\n" 
            "- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
            "- The generated query should return all of the information asked in the question without any missing or extra information.\n"
            "- Before generating the final SQL query, please think through the steps of how to write the query.\n\n"
            "Output Format:\n"
            "In your answer, please enclose the generated SQL query in a code block:\n```sql\n-- Your SQL query\n```\n\n"
            "Take a deep breath and think step by step to find the correct SQL query.\n"
        )

        return prompt
    
    def omni_sql_non_cot_prompt(self, question, schema):
        prompt = (
            "Task Overview:\n"
            "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.\n\n"
            "Database Engine:\n"
            "SQLite\n\n"
            "Database Schema:\n"
            f"{schema}\n"
            "This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.\n\n"
            "Question:\n"
            f"{question}\n\n"
            # "Instructions:\n" 
            # "- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
            # "- The generated query should return all of the information asked in the question without any missing or extra information.\n"
            # "- Before generating the final SQL query, please think through the steps of how to write the query.\n\n"
            "Output Format:\n"
            "In your answer, please enclose the generated SQL query in a code block:\n```sql\n-- Your SQL query\n```\n\n"
            "Take a deep breath and think step by step to find the correct SQL query.\n"
        )

        return prompt

class Text2SQLCotPrompt:
    def __init__(self):
        pass

    def text2sql_cot_prompt(self, schema, question, sql):
        prompt = f"""
            You are a senior data analyst specializing in SQL. Your task is to translate a natural language question into an executable SQLite query, providing a detailed reasoning trace.

            You will also receive a reference solution from a colleague, which may or may not be correct. This extra information intends to help you generate your answer, but you are asked not to mention the reference solution in any form.
            The reference solution might include: 
            1. Unnecessary table and column selections. 
            2. Incorrect or excessive joins. 
            3. Misalignment with the question.
            4. Opportunities for simplification.

            Ensure the SQL query is presented in a Markdown code block with proper syntax highlighting, like this:
            ```sql
            SELECT * FROM table;
            ```

            [Database Schema]:
            {schema}

            [Natural Language Question]:
            {question}

            [Reference Solution]:
            ```sql
            {sql}
            ```

            Provide your step-by-step text-to-SQL solution here.
        """
        return prompt
    
    def text2sql_cot_prompt_backup(self, schema, question, sql):
        template = """You are a senior data analyst who specializes in solving complex data query problems using SQL. Your task is to **reason step-by-step from a natural language question to its corresponding SQL query**, based on the provided database schema, question, and SQL statement. What I need is the reasoning process.
        Please present your thought process clearly and systematically. This should include (but not be limited to) the following aspects:
        1. What are the key pieces of information mentioned in the question?
        2. From which tables should the data be retrieved?
        3. Which fields or columns are involved?
        4. Are there operations such as aggregation, filtering, or sorting required?
        5. Why was the SQL written this way? Explain the logic behind each step.
        Your final output should be about how you arrived at the SQL query from the original question.
        [Database Schema]:
        {schema}
        [Natural Language Question]:
        {question}
        [SQL]:
        ```sql
        {sql}
        ```
        Please provide your step-by-step analysis. Begin with let's think step by step."""
        return template.format(schema=schema, question=question, sql=sql)
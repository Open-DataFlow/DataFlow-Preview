import sys
from dataflow.utils.registry import LazyLoader

from .Reasoning import *
<<<<<<< Updated upstream
from .Text2SQL import *

# from .KnowledgeCleaning import *
=======
from .KnowledgeCleaning import *
>>>>>>> Stashed changes
from .AgenticRAG import *


cur_path = "dataflow/operators/generate/"
_import_structure = {
    # reasoning
    "AnswerGenerator": (cur_path + "Reasoning/AnswerGenerator.py", "AnswerGenerator"),
    "QuestionCategoryClassifier": (cur_path + "Reasoning/QuestionCategoryClassifier.py", "QuestionCategoryClassifier"),
    "QuestionDifficultyClassifier": (cur_path + "Reasoning/QuestionDifficultyClassifier.py", "QuestionDifficultyClassifier"),
    "QuestionGenerator": (cur_path + "Reasoning/QuestionGenerator.py", "QuestionGenerator"),
    "AnswerExtraction_QwenMathEval": (cur_path + "Reasoning/AnswerExtraction_QwenMathEval.py", "AnswerExtraction_QwenMathEval"),
    "PseudoAnswerGenerator": (cur_path + "Reasoning/PseudoAnswerGenerator.py", "PseudoAnswerGenerator"),
    # text2sql
    "DatabaseSchemaExtractor": (cur_path + "Text2SQL/DatabaseSchemaExtractor.py", "DatabaseSchemaExtractor"),
    "ExtraKnowledgeGenerator": (cur_path + "Text2SQL/ExtraKnowledgeGenerator.py", "ExtraKnowledgeGenerator"),
    "PromptGenerator": (cur_path + "Text2SQL/PromptGenerator.py", "PromptGenerator"),
    "QuestionRefiner": (cur_path + "Text2SQL/QuestionRefiner.py", "QuestionRefiner"),
    "SchemaLinking": (cur_path + "Text2SQL/SchemaLinking.py", "SchemaLinking"),
    "SQLDifficultyClassifier": (cur_path + "Text2SQL/SQLDifficultyClassifier.py", "SQLDifficultyClassifier"),
    "SQLFilter": (cur_path + "Text2SQL/SQLFilter.py", "SQLFilter"),
    "Text2SQLDifficultyClassifier": (cur_path + "Text2SQL/Text2SQLDifficultyClassifier.py", "Text2SQLDifficultyClassifier"),
    # KBC
    "CorpusTextSplitter": (cur_path + "KnowledgeCleaning/CorpusTextSplitter.py", "CorpusTextSplitter"),
    "KnowledgeExtractor": (cur_path + "KnowledgeCleaning/KnowledgeExtractor.py", "KnowledgeExtractor"),
    "KnowledgeCleaner": (cur_path + "KnowledgeCleaning/KnowledgeCleaner.py", "KnowledgeCleaner"),
    "MultiHopQAGenerator": (cur_path + "KnowledgeCleaning/MultiHopQAGenerator.py", "MultiHopQAGenerator"),
    "AutoPromptGenerator": (cur_path + "AgenticRAG/AutoPromptGenerator.py", "AutoPromptGenerator"),
    "QAScorer": (cur_path + "AgenticRAG/QAScorer.py", "QAScorer"),
    "QAGenerator": (cur_path + "AgenticRAG/QAGenerator.py", "QAGenerator"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/generate/", _import_structure)

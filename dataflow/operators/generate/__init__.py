import sys
from dataflow.utils.registry import LazyLoader
from .Reasoning import *
#from .KnowledgeCleaning import *
from .AgenticRAG import *

cur_path = "dataflow/operators/generate/"
_import_structure = {
    "AnswerGenerator": (cur_path + "Reasoning/AnswerGenerator.py", "AnswerGenerator"),
    "QuestionCategoryClassifier": (cur_path + "Reasoning/QuestionCategoryClassifier.py", "QuestionCategoryClassifier"),
    "QuestionDifficultyClassifier": (cur_path + "Reasoning/QuestionDifficultyClassifier.py", "QuestionDifficultyClassifier"),
    "QuestionGenerator": (cur_path + "Reasoning/QuestionGenerator.py", "QuestionGenerator"),
    "AnswerExtraction_QwenMathEval": (cur_path + "Reasoning/AnswerExtraction_QwenMathEval.py", "AnswerExtraction_QwenMathEval"),
    "PseudoAnswerGenerator": (cur_path + "Reasoning/PseudoAnswerGenerator.py", "PseudoAnswerGenerator"),
    "CorpusTextSplitter": (cur_path + "KnowledgeCleaning/CorpusTextSplitter.py", "CorpusTextSplitter"),
    "KnowledgeExtractor": (cur_path + "KnowledgeCleaning/KnowledgeExtractor.py", "KnowledgeExtractor"),
    "KnowledgeCleaner": (cur_path + "KnowledgeCleaning/KnowledgeCleaner.py", "KnowledgeCleaner"),
    "AutoPromptGenerator": (cur_path + "AgenticRAG/AutoPromptGenerator.py", "AutoPromptGenerator"),
    "QAScorer": (cur_path + "AgenticRAG/QAScorer.py", "QAScorer"),
    "QAGenerator": (cur_path + "AgenticRAG/QAGenerator.py", "QAGenerator"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/generate/", _import_structure)

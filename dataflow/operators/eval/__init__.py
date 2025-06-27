import sys
from dataflow.utils.registry import LazyLoader

cur_path = "dataflow/operators/eval/"
_import_structure = {  
    "NgramScorer": (cur_path + "statistics/ngram_scorer.py", "NgramScorer"),
    "LexicalDiversityScorer": (cur_path + "statistics/lexical_diversity_scorer.py", "LexicalDiversityScorer"),
    "LangkitScorer": (cur_path + "statistics/langkit_scorer.py", "LangkitScorer"),
    
    "DeitaQualityScorer": (cur_path + "models/deita_quality_scorer.py", "DeitaQualityScorer"),
    "InstagScorer": (cur_path + "models/instag_scorer.py", "InstagScorer"),
    "DebertaV3Scorer": (cur_path + "models/debertav3_scorer.py", "DebertaV3Scorer"),
    "DeitaComplexityScorer": (cur_path + "models/deita_complexity_scorer.py", "DeitaComplexityScorer"),
    "FineWebEduScorer": (cur_path + "models/fineweb_edu_scorer.py", "FineWebEduScorer"),
    "PairQualScorer": (cur_path + "models/pair_qual_scorer.py", "PairQualScorer"),
    "PresidioScorer": (cur_path + "models/presidio_scorer.py", "PresidioScorer"),
    "RMScorer": (cur_path + "models/rm_scorer.py", "RMScorer"),
    "TextbookScorer": (cur_path + "models/textbook_scorer.py", "TextbookScorer"),
    "SuperfilteringScorer": (cur_path + "models/superfiltering_scorer.py", "SuperfilteringScorer"),
    "QuratingScorer": (cur_path + "models/qurating_scorer.py", "QuratingScorer"),
    "PerplexityScorer": (cur_path + "models/perplexity_scorer.py", "PerplexityScorer"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/eval/", _import_structure)

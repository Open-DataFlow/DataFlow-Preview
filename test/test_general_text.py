 
from dataflow.operators.process.GeneralText import FineWebEduFilter, PairQualFilter, QuratingFilter, TextbookFilter

from dataflow.utils.storage import FileStorage

class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = '../dataflow_cache'
        self.quality_filter1 = PairQualFilter(min_score=2.5, max_score=10000, lang='en', model_cache_dir=self.model_cache_dir)
        self.quality_filter2 = FineWebEduFilter(min_score=2.5, max_score=10000, model_cache_dir=self.model_cache_dir, device='cuda')
        self.quality_filter3 = QuratingFilter(model_cache_dir=self.model_cache_dir)
    def forward(self):
        self.quality_filter1.run(
            storage = self.storage.step(),
            input_key = "raw_content"
        )
        self.quality_filter2.run(
            storage = self.storage.step(),
            input_key = "raw_content"
        )
        self.quality_filter3.run(
            storage = self.storage.step(),
            input_key = "raw_content"
        )

model = TextPipeline()
model.forward()

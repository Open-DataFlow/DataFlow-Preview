from dataflow.operators.process.GeneralText import NgramFilter, MinHashDeduplicator, QuratingFilter
from dataflow.operators.refine.GeneralText import HtmlUrlRemoverRefiner
from dataflow.utils.storage import FileStorage

class TextPipeline():
    def __init__(self):
        
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        self.qrt = QuratingFilter()
        
    def forward(self):
        
        self.qrt.run(
            storage = self.storage.step(),
            input_key = "raw_content"
        )

model = TextPipeline()
model.forward()
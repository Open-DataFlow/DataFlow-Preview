from dataflow.operators.process.GeneralText import DeitaComplexityFilter

from dataflow.utils.storage import FileStorage

class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/sft_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.scorer1 = DeitaComplexityFilter(model_cache_dir=self.model_cache_dir)

    def forward(self):
        self.scorer1.run(
            storage = self.storage.step()
        )
if __name__ == "__main__":
    model = TextPipeline()
    model.forward()

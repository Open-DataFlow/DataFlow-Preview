from dataflow.operators.process.GeneralText import (
    WordNumberFilter,
    SuperfilteringFilter,
    DeitaQualityFilter,
    InstagFilter
)
from dataflow.utils.storage import FileStorage


class SFTTextPipeline():
    
    def __init__(self):
        
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/sft_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        self.model_cache_dir = '/mnt/public/code/zzy/dataflow_cache'
        self.word_number_filter_step1 = WordNumberFilter(
            min_words=20,
            max_words=1000
        )
        self.super_filtering_filter_step2 = SuperfilteringFilter(
            min_score=0.5,
            max_score=10000.0,
            use_API=False,
            model_name="/mnt/public_2/code/zzy/dataflow_cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
            max_length=512,
        )
        self.deita_quality_filter_step3 = DeitaQualityFilter(
            min_score=2.5,
            max_score=10000,
            use_API=False,
            model_name="/mnt/public_2/code/zzy/dataflow_cache/models--hkust-nlp--deita-quality-scorer/snapshots/f30fe3e0cff454fd1f02953fd2b8bc3d1e67697d",
            max_length=512,
        )
        
        self.instag_filter_step4 = InstagFilter(
            min_score=2,
            max_score=10000,
            use_API=False,
            model_name="/mnt/public_2/code/zzy/dataflow_cache/models--OFA-Sys--InsTagger/snapshots/261bb8900245774471bc04421ddd47930a0bd28a",
            max_new_tokens=1024,
            do_sample=False,
            num_return_sequences=1,
            return_dict_in_generate=True,
            temperature=0
        )
        
    def forward(self):
        
        self.word_number_filter_step1.run(
            storage=self.storage.step(),
            input_key="output",
        )
        
        self.super_filtering_filter_step2.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_input_key=None,
            input_output_key='output'
        )
        
        self.deita_quality_filter_step3.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_output_key='output'
        )
        
        self.instag_filter_step4.run(
            storage=self.storage.step(),
            input_instruction_key='instruction'
        )
        
pipeline = SFTTextPipeline()
pipeline.forward()
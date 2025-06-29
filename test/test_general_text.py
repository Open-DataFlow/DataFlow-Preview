 
from dataflow.operators.process.GeneralText import (
    MinHashDeduplicator,
    ColonEndFilter,
    WordNumberFilter,
    BlocklistFilter,
    SentenceNumberFilter
)
from dataflow.operators.refine.GeneralText import (
    HtmlUrlRemoverRefiner,
    RemoveEmojiRefiner,
    RemoveExtraSpacesRefiner
)
from dataflow.utils.storage import FileStorage

class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.remove_extra_spaces_refiner = RemoveExtraSpacesRefiner()
        self.remove_emoji_refiner = RemoveEmojiRefiner()
        self.html_remove_refiner = HtmlUrlRemoverRefiner()
        self.minhash_deduplicator = MinHashDeduplicator(num_perm=128, threshold=0.9, use_n_gram=True, ngram=5)
        self.blocklist_filter = BlocklistFilter()
        self.word_number_filter = WordNumberFilter(min_words=20, max_words=100000)
        self.colon_end_filter = ColonEndFilter()
        self.sentence_number_filter = SentenceNumberFilter(min_sentences=3, max_sentences=7500)

    def forward(self):
        self.remove_extra_spaces_refiner.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.remove_emoji_refiner.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.html_remove_refiner.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.minhash_deduplicator.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.blocklist_filter.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.word_number_filter.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.colon_end_filter.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.sentence_number_filter.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )

model = TextPipeline()
model.forward()

from dataflow.operators.generate.AgenticRAG import (
   AutoPromptGenerator,
   QAGenerator,
   QAScorer
)

from dataflow.operators.process.AgenticRAG import (
    ContentChooser,
)
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request

class AgenticRAGPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/AgenticRAGPipeline/pipeline_small_chunk.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        api_llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )

        self.content_chooser_step1 = ContentChooser(embedding_model_path="/mnt/public/data/lh/models/hub/gte-Qwen2-7B-instruct")

        self.prompt_generator_step2 = AutoPromptGenerator(api_llm_serving)

        self.qa_generator_step3 = QAGenerator(api_llm_serving)

        self.qa_scorer_step4 = QAScorer(api_llm_serving)
        
        # 未来或许可以维护一个类似nn.sequential的容器，方便添加并实例化多个算子
    def forward(self):

        self.content_chooser_step1.run(
            storage = self.storage.step(),
            input_key= "text",
            num_samples=5,
            method= "random"
        )

        self.prompt_generator_step2.run(
            storage = self.storage.step(),
            input_key = "text"
        )

        self.qa_generator_step3.run(
            storage = self.storage.step(),
            input_key="text",
            prompt_key="generated_prompt",
            output_quesion_key="generated_question",
            output_answer_key="generated_answer"
        )

        self.qa_scorer_step4.run(
            storage = self.storage.step(),
            input_question_key="generated_question",
            input_answer_key="generated_answer",
            output_question_quality_key="question_quality_grades",
            output_question_quality_feedback_key="question_quality_feedbacks",
            output_answer_alignment_key="answer_alignment_grades",
            output_answer_alignment_feedback_key="answer_alignment_feedbacks",
            output_answer_verifiability_key="answer_verifiability_grades",
        )
        
model = AgenticRAGPipeline()
model.forward()


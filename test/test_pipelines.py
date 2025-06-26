import pytest
from test_reasoning import ReasoningPipeline
from test_reasoning_pretrain import ReasoningPipeline_Pretrain
from test_general_text import TextPipeline
from test_agentic_rag import AgenticRAGPipeline
@pytest.fixture(scope="session")
def llm_serving():
    from dataflow.llmserving import LocalModelLLMServing
    return LocalModelLLMServing(
        model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        tensor_parallel_size=4,
        max_tokens=8192,
        max_model_len=8192,
        model_source="local"
    )

@pytest.mark.gpu
def test_reasoning_pipeline(llm_serving): 
    pytest
    resoning_pipe = ReasoningPipeline(llm_serving=llm_serving)
    resoning_pipe.forward()

@pytest.mark.gpu
def test_text_pipeline(llm_serving):
    text_pipe = TextPipeline(llm_serving=llm_serving)
    text_pipe.forward()

@pytest.mark.gpu
def test_agentic_rag_pipeline(llm_serving):
    rag_pipe = AgenticRAGPipeline(llm_serving=llm_serving)
    rag_pipe.forward()

@pytest.mark.gpu
def test_reasoning_pipeline_pretrain(llm_serving):
    reasoning_pipe_pretrain = ReasoningPipeline_Pretrain(llm_serving=llm_serving)
    reasoning_pipe_pretrain.forward()
import logging
from vllm import LLM,SamplingParams
from huggingface_hub import snapshot_download
from dataflow.core import LLMServingABC

class LocalModelLLMServing(LLMServingABC):
    '''
    A class for generating text using vllm, with model from huggingface or local directory
    '''
    def __init__(self, 
                 tensor_parallel_size: int = 1,
                 model_name_or_path: str = None,
                 cache_dir: str = None,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_tokens: int = 1024,
                 top_k: int = 40,
                 repetition_penalty: float = 1.0,
                 seed: int = 42,
                 download_dir: str = "./ckpt/models/",
                 max_model_len: int = 4096,
                 model_source: str= "remote",
                 ):

        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        if(model_source=="local"):
            self.real_model_path = model_name_or_path
        elif(model_source=="remote"):  
            try:
                self.real_model_path = snapshot_download(
                    repo_id=model_name_or_path,
                    cache_dir=cache_dir,
                    local_dir=f"{download_dir}{model_name_or_path}",
                )
            except:
                self.real_model_path = model_name_or_path
        logging.info(f"Model will be loaded from {self.real_model_path}")
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed
            # hat_template_kwargs={"enable_thinking": False},
        )
        self.llm = LLM(
            model=self.real_model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
    def generate(self):
        # # read input file : accept jsonl file only
        # dataframe = self.datastorage.read(self.input_file, "dataframe")
        # # check if input_key are in the dataframe
        # if self.input_key not in dataframe.columns:
        #     key_list = dataframe.columns.tolist()
        #     raise ValueError(f"input_key: {self.input_key} not found in the dataframe, please check the input_key: {key_list}")
        # # check if output_key are in the dataframe
        # if self.output_key in dataframe.columns:
        #     key_list = dataframe.columns.tolist()
        #     raise ValueError(f"Found {self.output_key} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        # # generate text
        # user_prompts = dataframe[self.input_key].tolist()
        # full_prompts = [self.system_prompt + '\n' + user_prompt for user_prompt in user_prompts]
        # responses = self.llm.generate(full_prompts, self.sampling_params)
        # dataframe[self.output_key] = [output.outputs[0].text for output in responses]
        # self.datastorage.write(self.output_file, dataframe)
        # return
        pass
    
    
    def generate_from_input(self, 
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:
        full_prompts = [system_prompt + '\n' + question for question in user_inputs]
        responses = self.llm.generate(full_prompts, self.sampling_params)
        return [output.outputs[0].text for output in responses]
    
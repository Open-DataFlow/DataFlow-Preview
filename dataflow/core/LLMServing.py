from abc import ABC, abstractmethod
from typing import Any, List


class LLMServingABC(ABC):
    """Abstract base class for data generators. Which may be used to generate data from a model or API. Called by operators
    """
    @abstractmethod
    def generate(self) -> Any:
        """
        Generate data from config of the generator.
        config: dict, include things like model_name, input_file, output_file, etc.
        """
        pass
    
    @abstractmethod
    def generate_from_input(self, user_inputs: List[str], system_prompt: str) -> List[str]:
        """
        Generate data from input.
        input: List[str], the input of the generator
        """
        pass
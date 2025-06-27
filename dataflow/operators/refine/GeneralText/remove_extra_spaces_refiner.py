import re
from tqdm import tqdm
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class RemoveExtraSpacesRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(self, lang):
        return "去除文本中的多余空格" if lang == "zh" else "Remove extra spaces in the text."

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []

        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            refined_text = " ".join(original_text.split())  # Remove extra spaces

            if original_text != refined_text:
                item = refined_text
                modified = True
                self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {refined_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")

        dataframe[self.input_key] = refined_data
        storage.write(dataframe)
        self.logger.info(f"Refining completed. Total items modified: {numbers}")

        return [self.input_key]

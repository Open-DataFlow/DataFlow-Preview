from abc import ABC, abstractmethod
import pandas as pd
import json
from typing import Any, Literal
import os


class DataFlowStorage(ABC):
    """
    Abstract base class for data storage.
    """
    @abstractmethod
    def read(self, file_path: Any, type) -> Any:
        """
        Read data from file.
        file_path: str, the path of the file to read
        type: type that you want to read to, such as "datatrame", List[dict], etc.
        """
        pass
    
    @abstractmethod
    def write(self, file_path: Any, data: Any) -> Any:
        pass

class FileStorage(DataFlowStorage):
    """
    Storage for file system.
    """
    def __init__(self, config: dict):
        self.config = config
        #if "input_file" not in self.config or "output_file" not in self.config:
        #    raise ValueError("input/output_file are required")
        #self.input_file = self.config["input_file"]
        #self.output_file = self.config["output_file"]
        #if self.input_file == None or self.input_file == "" or self.output_file == None or self.output_file == "":
        #    raise ValueError("input/output_file are required")
    
    def read(self, file_path: str, type: Literal["dataframe", "dict"]) -> Any:
        """
        Read data from file.
        file_path: str, the path of the file to read, it should end with json, jsonl, csv, parquet, pickle
        type: type that you want to read to, such as "datatrame", List[dict], etc.
        """
        # make file directory
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)

        # load data from file
        ends = file_path.split(".")[-1]
        if ends == "json":
            dataframe = pd.read_json(file_path)
        elif ends == "jsonl":
            dataframe = pd.read_json(file_path, lines=True)
        elif ends == "csv":
            dataframe = pd.read_csv(file_path)
        elif ends == "parquet":
            dataframe = pd.read_parquet(file_path)
        elif ends == "pickle":
            dataframe = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ends}, input file should end with json, jsonl, csv, parquet, pickle")
        
        if type == "dataframe":
            return dataframe
        elif type == "dict":
            return dataframe.to_dict(orient="records")
        raise ValueError(f"Unsupported type: {type}, type should be dataframe or dict")
        
    def write(self, file_path: str, data: Any) -> Any:
        """
        Write data to file.
        file_path: str, the path of the file to write, it should end with json, jsonl, csv, parquet, pickle
        data: Any, the data to write, it should be a dataframe, List[dict], etc.
        """
        if type(data) == list:
            if type(data[0]) == dict:
                dataframe = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data[0])}")
        elif type(data) == pd.DataFrame:
            dataframe = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        ends = file_path.split(".")[-1]
        if ends == "json":
            dataframe.to_json(file_path, orient="records", force_ascii=False)
        elif ends == "jsonl":
            dataframe.to_json(file_path, orient="records", lines=True, force_ascii=False)
        elif ends == "csv":
            dataframe.to_csv(file_path, index=False)
        elif ends == "parquet":
            dataframe.to_parquet(file_path)
        elif ends == "pickle":
            dataframe.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ends}, output file should end with json, jsonl, csv, parquet, pickle")
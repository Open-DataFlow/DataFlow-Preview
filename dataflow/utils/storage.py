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
    def read(self, output_type) -> Any:
        """
        Read data from file.
        type: type that you want to read to, such as "datatrame", List[dict], etc.
        """
        pass
    
    @abstractmethod
    def write(self, data: Any) -> Any:
        pass

class FileStorage(DataFlowStorage):
    """
    Storage for file system.
    """
    def __init__(self, 
                 first_entry_file_name: str,
                 cache_path:str="./cache",
                 file_name_prefix:str="dataflow_cache_step",
                 cache_type:Literal["json", "jsonl", "csv", "parquet", "pickle"] = "jsonl"
                 ):
        self.first_entry_file_name = first_entry_file_name
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        self.operator_step = -1

    def _get_cache_file_path(self, step) -> str:
        if step == 0:
            # If it's the first step, use the first entry file name
            return os.path.join(self.first_entry_file_name)
        else:
            return os.path.join(self.cache_path, f"{self.file_name_prefix}_{step}.{self.cache_type}")

    def step(self):
        self.operator_step += 1
        return self
    
    def reset(self):
        self.operator_step = -1
        return self
    
    def _load_local_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load data from local file based on file type."""
        try:
            if file_type == "json":
                return pd.read_json(file_path)
            elif file_type == "jsonl":
                return pd.read_json(file_path, lines=True)
            elif file_type == "csv":
                return pd.read_csv(file_path)
            elif file_type == "parquet":
                return pd.read_parquet(file_path)
            elif file_type == "pickle":
                return pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Failed to load {file_type} file: {str(e)}")
    
    def _convert_output(self, dataframe: pd.DataFrame, output_type: str) -> Any:
        """Convert dataframe to requested output type."""
        if output_type == "dataframe":
            return dataframe
        elif output_type == "dict":
            return dataframe.to_dict(orient="records")
        raise ValueError(f"Unsupported output type: {output_type}")

    def read(self, output_type: Literal["dataframe", "dict"]) -> Any:
        """
        Read data from current file managed by storage.
        
        Args:
            output_type: Type that you want to read to, either "dataframe" or "dict".
            Also supports remote datasets with prefix:
                - "hf:{dataset_name}{:config}{:split}"  => HuggingFace dataset eg. "hf:openai/gsm8k:main:train"
                - "ms:{dataset_name}{}:split}"          => ModelScope dataset eg. "ms:modelscope/gsm8k:train"
        
        Returns:
            Depending on output_type:
            - "dataframe": pandas DataFrame
            - "dict": List of dictionaries
        
        Raises:
            ValueError: For unsupported file types or output types
        """
        file_path = self._get_cache_file_path(self.operator_step)
        print(f"Reading data from {file_path} with type {output_type}")

        if self.operator_step == 0:
            source = self.first_entry_file_name
            print(f"Reading remote dataset from {source} with type {output_type}")
            if source.startswith("hf:"):
                from datasets import load_dataset
                _, dataset_name, *parts = source.split(":")

                if len(parts) == 1:
                    config, split = None, parts[0]
                elif len(parts) == 2:
                    config, split = parts
                else:
                    config, split = None, "train"

                dataset = (
                    load_dataset(dataset_name, config, split=split) 
                    if config 
                    else load_dataset(dataset_name, split=split)
                )
                dataframe = dataset.to_pandas()
                return self._convert_output(dataframe, output_type)
        
            elif source.startswith("ms:"):
                from modelscope import MsDataset
                _, dataset_name, *split_parts = source.split(":")
                split = split_parts[0] if split_parts else "train"

                dataset = MsDataset.load(dataset_name, split=split)
                dataframe = pd.DataFrame(dataset)
                return self._convert_output(dataframe, output_type)
                            
            else:
                local_cache = file_path.split(".")[-1]
        else:
            local_cache = self.cache_type

        dataframe = self._load_local_file(file_path, local_cache)
        return self._convert_output(dataframe, output_type)
        
    def write(self, data: Any) -> Any:
        """
        Write data to current file managed by storage.
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

        file_path = self._get_cache_file_path(self.operator_step + 1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Writing data to {file_path} with type {self.cache_type}")
        if self.cache_type == "json":
            dataframe.to_json(file_path, orient="records", force_ascii=False, indent=2)
        elif self.cache_type == "jsonl":
            dataframe.to_json(file_path, orient="records", lines=True, force_ascii=False)
        elif self.cache_type == "csv":
            dataframe.to_csv(file_path, index=False)
        elif self.cache_type == "parquet":
            dataframe.to_parquet(file_path)
        elif self.cache_type == "pickle":
            dataframe.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.cache_type}, output file should end with json, jsonl, csv, parquet, pickle")
        
        return file_path
    

class DBStorage(DataFlowStorage):
    """
    Storage for database.
    This is a placeholder class, you can implement your own database storage.
    """
    def __init__(self, db_config: dict):
        self.db_config = db_config

    def read(self, output_type: Literal["dataframe", "dict"]) -> Any:
        raise NotImplementedError("DBStorage.read() is not implemented yet.")
    
    def excute_read(self, expr:str, output_type: Literal["dataframe", "dict"]) -> Any:
        pass

    def write(self, data: Any) -> Any:
        raise NotImplementedError("DBStorage.write() is not implemented yet.")
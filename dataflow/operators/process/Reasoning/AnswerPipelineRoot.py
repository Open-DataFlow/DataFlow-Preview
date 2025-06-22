# 根节点，用来将数据拆入不同的分支
import pandas as pd
from dataflow.utils.utils import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.reasoning_utils.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor
from dataflow.utils.Operator import Operator
from dataflow.utils.Storage import FileStorage


@OPERATOR_REGISTRY.register()
class AnswerPipelineRoot(Operator):
    def __init__(self, config: dict):
        self.check_config(config)
        self.config = config
        self.input_file = config.get("input_file")
        self.output_file_with_gt = config.get("output_file_with_gt")
        self.output_file_without_gt = config.get("output_file_without_gt")
        self.input_answer_key = config.get("input_answer_key")
        self.input_gt_key = config.get("input_gt_key", "")
        self.logger = get_logger()
        self.datastorage = FileStorage(config)

    def check_config(self, config: dict) -> None:
        required_keys = ['input_file', 'output_file_with_gt', 'output_file_without_gt']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "答案处理流程根节点，负责将输入数据根据有无真实标签GT分发到不同处理分支。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- output_dir：输出目录路径\n"
                "- branch_config：分支配置参数\n"
                "- parallel_workers：并行工作线程数\n\n"
                "输出参数：\n"
                "- 多个输出文件路径（根据分支配置生成）"
            )
        elif lang == "en":
            return (
                "Root node of answer processing pipeline, distributes input data to different processing branches.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- output_dir: Output directory path\n"
                "- branch_config: Branch configuration parameters\n"
                "- parallel_workers: Number of parallel workers\n\n"
                "Output Parameters:\n"
                "- Multiple output file paths (generated based on branch config)"
            )
        else:
            return "AnswerPipelineRoot routes data to different processing branches."

    def run(self):
        df = self.datastorage.read(self.input_file, "dataframe")
        
        if not self.input_gt_key or self.input_gt_key not in df.columns:
            self.logger.warning("No valid gt key in input file, copy input file to output file without gt")
            return
            
        # 初始化答案提取器
        if self.input_answer_key in df.columns:
            unit_text_manager = UnitTextManager()
            string_cleaner = StringCleaner(unit_text_manager)
            answer_extractor = AnswerExtractor(string_cleaner)
        
            def extract_gt(answer, gt):
                try:
                    if gt != "" and not pd.isna(gt):
                        return gt
                    else:
                        if pd.isna(answer) or answer == "":
                            return None
                        else:
                            return answer_extractor.extract_answer(answer,None,True)
                except Exception as e:
                    self.logger.error(f"Error in extract_gt: {e}", exc_info=True)
                    return None
            
            # 使用 apply 遍历 DataFrame, 避免显式循环索引问题
            df[self.input_gt_key] = df.apply(lambda row: extract_gt(row[self.input_answer_key],
                                                                    row[self.input_gt_key]),
                                            axis=1)
        
        # 拆分有gt和无gt的 DataFrame
        df_with_gt = df[(df[self.input_gt_key].notna()) & (df[self.input_gt_key] != "")]
        df_without_gt = df[(df[self.input_gt_key].isna()) | (df[self.input_gt_key] == "")].copy()
        df_without_gt[self.input_gt_key] = None

        
        # 输出结果
        if len(df_with_gt) > 0:
            self.datastorage.write(self.output_file_with_gt, df_with_gt)
        if len(df_without_gt) > 0:
            self.datastorage.write(self.output_file_without_gt, df_without_gt)
        self.logger.info(f"output {df_with_gt.shape[0]} rows with gt to {self.output_file_with_gt}")
        self.logger.info(f"output {df_without_gt.shape[0]} rows without gt to {self.output_file_without_gt}")
                    
            



        
        
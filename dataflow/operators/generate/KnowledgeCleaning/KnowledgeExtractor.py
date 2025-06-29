import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.utils.kbcleaning import _parse_pdf_to_md,_parse_doc_to_md,_parse_xml_to_md
import os

@OPERATOR_REGISTRY.register()
class KnowledgeExtractor(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, **kwargs):
        self.logger = get_logger()
        self.intermediate_dir=kwargs.get("intermediate_dir", "intermediate")
        
    @staticmethod
    def get_desc(self, lang="en"):
        """
        返回算子功能描述 (根据run()函数的功能实现)
        """
        if lang == "zh":
            return (
                "知识提取算子：支持从多种文件格式中提取结构化内容并转换为标准Markdown\n"
                "核心功能：\n"
                "1. PDF文件：使用MinerU解析引擎提取文本/表格/公式，保留原始布局\n"
                "2. Office文档(DOC/PPT等)：通过DocConverter转换为Markdown格式\n"
                "3. 网页内容(HTML/XML)：使用trafilatura提取正文并转为Markdown\n"
                "4. 纯文本(TXT/MD)：直接透传不做处理\n"
                "特殊处理：\n"
                "- 自动识别中英文文档(lang参数)\n"
                "- 支持本地文件路径和URL输入\n"
                "- 生成中间文件到指定目录(intermediate_dir)"
            )
        else:  # 默认英文
            return (
                "Knowledge Extractor: Converts multiple file formats to structured Markdown\n"
                "Key Features:\n"
                "1. PDF: Uses MinerU engine to extract text/tables/formulas with layout preservation\n"
                "2. Office(DOC/PPT): Converts to Markdown via DocConverter\n"
                "3. Web(HTML/XML): Extracts main content using trafilatura\n"
                "4. Plaintext(TXT/MD): Directly passes through without conversion\n"
                "Special Handling:\n"
                "- Auto-detects Chinese/English documents(lang param)\n"
                "- Supports both local files and URLs\n"
                "- Generates intermediate files to specified directory(intermediate_dir)"
            )

    def run(self, storage:DataFlowStorage ,raw_file=None, url=None,lang="en"):
        self.logger.info("starting to extract...")
        self.logger.info("If you are providing a url or a large file, this may take a while, please wait...")
        if(url):
            output_file=os.path.join(os.path.dirname(storage.first_entry_file_name), "raw/crawled.md")
            output_file=_parse_xml_to_md(url=url,output_file=output_file)
            self.logger.info(f"Primary extracted result written to: {output_file}")
            return output_file

        raw_file_name=os.path.splitext(os.path.basename(raw_file))[0]
        raw_file_suffix=os.path.splitext(raw_file)[1]
        raw_file_suffix_no_dot=raw_file_suffix.replace(".","")
        output_file=os.path.join(self.intermediate_dir,f"{raw_file_name}_{raw_file_suffix_no_dot}.md")
        if(raw_file_suffix==".pdf"):
            try:
                from mineru.data.data_reader_writer import FileBasedDataWriter
                from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
                from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
                from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
                from mineru.utils.enum_class import MakeMode
            except:
                raise Exception(
                    """
MinerU is not installed in this environment yet.
Please refer to https://github.com/opendatalab/mineru to install.
Or you can just execute 'pip install mineru[pipeline]' and 'mineru-models-download' to fix this error.
please make sure you have gpu on your machine.
"""
                )
            # optional: 是否从本地加载OCR模型
            os.environ['MINERU_MODEL_SOURCE'] = "local"
            output_file=_parse_pdf_to_md(
                raw_file,
                self.intermediate_dir,
                lang,
                "txt"
            )
        elif(raw_file_suffix in [".doc", ".docx", ".pptx", ".ppt"]):
            try:
                from magic_doc.docconv import DocConverter
            except:
                raise Exception(
                    """
Fairy-doc is not installed in this environment yet.
Please refer to https://github.com/opendatalab/magic-doc to install.
Or you can just execute 'apt-get/yum/brew install libreoffice' and 'pip install fairy-doc[gpu]' to fix this error.
please make sure you have gpu on your machine.
"""
                )
            if(raw_file_suffix==".docx"):
                raise Exception("Function Under Maintaining...Please try .doc format file instead.")
            output_file=_parse_doc_to_md(raw_file, output_file)
        elif(raw_file_suffix in [".html", ".xml"]):
            output_file=_parse_xml_to_md(raw_file=raw_file,output_file=output_file)
        elif(raw_file_suffix in [".txt",".md"]):
            # for .txt and .md file, no action is taken
            output_file=raw_file
        else:
            raise Exception("Unsupported file type: " + raw_file_suffix)
        
        self.logger.info(f"Primary extracted result written to: {output_file}")
        return output_file


import os
from pathlib import Path
from trafilatura import fetch_url, extract
from dataflow.logger import get_logger

def _parse_pdf_to_md(
    input_pdf_path: str, 
    output_dir: str,      
    lang: str = "ch",     
    parse_method: str = "auto"  # 解析方法：auto/txt/ocr
):
    """
    将PDF转换为Markdown（仅使用Pipeline后端）
    """
    logger=get_logger()
    # 读取PDF文件
    pdf_bytes = Path(input_pdf_path).read_bytes()
    pdf_name = Path(input_pdf_path).stem

    # 解析PDF
    infer_results, all_image_lists, all_pdf_docs, _, ocr_enabled_list = pipeline_doc_analyze(
        [pdf_bytes], [lang], parse_method=parse_method
    )

    # 准备输出目录
    image_dir = os.path.join(output_dir, f"{pdf_name}_images")
    os.makedirs(image_dir, exist_ok=True)
    image_writer = FileBasedDataWriter(image_dir)
    md_writer = FileBasedDataWriter(output_dir)

    # 生成中间结果和Markdown
    middle_json = pipeline_result_to_middle_json(
        infer_results[0], all_image_lists[0], all_pdf_docs[0], 
        image_writer, lang, ocr_enabled_list[0], True
    )
    md_content = pipeline_union_make(middle_json["pdf_info"], MakeMode.MM_MD, os.path.basename(image_dir))
    # 保存Markdown
    md_writer.write_string(f"{pdf_name}_pdf.md", md_content)
    logger.info(f"Markdown saved to: {os.path.join(output_dir, f'{pdf_name}_pdf.md')}")

    return os.path.join(output_dir,f"{pdf_name}_pdf.md")

def _parse_doc_to_md(input_file: str, output_file: str):
    """
        support conversion of doc/ppt/pptx/pdf files to markdowns
    """
    logger=get_logger()
    converter = DocConverter(s3_config=None)
    markdown_content, time_cost = converter.convert(input_file, conv_timeout=300)
    logger.info("time cost: ", time_cost)
    with open(output_file, "w",encoding='utf-8') as f:
        f.write(markdown_content)
    return output_file

def _parse_xml_to_md(raw_file:str=None, url:str=None, output_file:str=None):
    logger=get_logger()
    if(url):
        downloaded=fetch_url(url)
    elif(raw_file):
        with open(raw_file, "r", encoding='utf-8') as f:
            downloaded=f.read()
    else:
        raise Exception("Please provide at least one of file path and url string.")

    try:
        result=extract(downloaded, output_format="markdown", with_metadata=True)
        logger.info(f"Extracted content is written into {output_file}")
        with open(output_file,"w", encoding="utf-8") as f:
            f.write(result)
    except Exception as e:
        logger.error("Error during extract this file or link: ", e)

    return output_file
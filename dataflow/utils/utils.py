import numpy as np
import subprocess
import torch
import logging
import colorlog

def get_operator(operator_name, args):
    from dataflow.utils.registry import OPERATOR_REGISTRY
    print(operator_name, args)
    operator = OPERATOR_REGISTRY.get(operator_name)(args)
    logger = get_logger()
    if operator is not None:
        logger.info(f"Successfully get operator {operator_name}, args {args}")
    else:
        logger.error(f"operator {operator_name} is not found")
    assert operator is not None
    print(operator)
    return operator

def get_logger(level=logging.INFO):
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(level)
    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # 定义颜色输出格式
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s %(asctime)s | %(filename)-20s- %(module)-20s- %(funcName)-20s- %(lineno)5d - %(name)-10s | %(levelname)8s | Processno %(process)5d - Threadno %(thread)-15d : %(message)s', 
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    # 将颜色输出格式添加到控制台日志处理器
    console_handler.setFormatter(color_formatter)
    # 移除默认的handler
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # 将控制台日志处理器添加到logger对象
    logger.addHandler(console_handler)
    return logger

def pipeline_step(yaml_path, step_name):
    import yaml
    logger = get_logger()
    logger.info(f"Loading yaml {yaml_path} ......")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config = merge_yaml(config)
    logger.info(f"Load yaml success, config: {config}")
    algorithm = get_operator(step_name, config)
    logger.info("Start running ...")
    algorithm.run()

def merge_yaml(config):
    if not config.get("vllm_used"):
        return config
    else:
        vllm_args_list = config.get("vllm_args", [])
        if isinstance(vllm_args_list, list) and len(vllm_args_list) > 0 and isinstance(vllm_args_list[0], dict):
            vllm_args = vllm_args_list[0]
            config.update(vllm_args)  # 合并进顶层
        return config
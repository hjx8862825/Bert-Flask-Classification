import logging

def my_logger(logging_path):
    # 生成日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.handlers = []
    assert len(logger.handlers) == 0
    handler = logging.FileHandler(logging_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
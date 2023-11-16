import logging
from config import load_config



def init_logger():
    logger = logging.getLogger("checkmatrix")

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(load_config().log_path)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def get_logger():
    logger = logging.getLogger("checkmatrix")

    return logger

import logging



logger = logging.getLogger('checkr')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/checkr.log')
stream_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

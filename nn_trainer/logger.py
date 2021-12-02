import logging

class Logger(object):
    def __init__(self, name: str, verbose:bool = True) -> None:

        # =============Logger===============
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.info('Initialized logger.')
        stream_handler = logging.StreamHandler()
        if verbose:
            stream_handler.setLevel(logging.DEBUG)
        else:
            stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler("model.log")
        self._logger.addHandler(stream_handler)
        self._logger.addHandler(file_handler)

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)
        
    def exception(self, msg):
        self._logger.exception(msg)

    def critical(self, msg):
        self._logger.critical(msg)
    
    def log(self, level:int, msg):
        self._logger.log(level, msg)

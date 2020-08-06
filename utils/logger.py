import logging
import os
from .utils import get_local_time


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    :param config
    example:
        logger = Logger(config)
        logger.debug(train_state)
        logger.info(train_result)
    """
    LOGROOT='./log/'
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfilename = '{}-{}.log'.format(config['model'], get_local_time())

    logger = logging.getLogger(logfilename)
    logger.setLevel(logging.DEBUG)

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fileformatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(sformatter)
    logger.addHandler(sh)

    return logger


mylogger = None


def get_logger(config=None):
    global mylogger
    if config is not None:
        mylogger = init_logger(config)
        return mylogger
    else:
        if mylogger is None:
            raise RuntimeError('logger must be initialized when the first usage!')
        else:
            return mylogger

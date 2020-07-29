import logging
import os
from .utils import get_local_time


class Logger(object):

    def __init__(self, config):
        """
        A logger that can show a message on standard output and write it into the
        file named `filename` simultaneously.
        All the message that you want to log MUST be str.

        :param config
        example:
            logger = Logger('train.log')
            logger.debug(train_state)
            logger.info(train_result)

        """

        self.config = config
        self.LOGROOT='./log/'
        dir_name = os.path.dirname(self.LOGROOT)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        logfilename = '{}-{}.log'.format(self.config['model'], get_local_time())

        self.logger = logging.getLogger(logfilename)
        self.logger.setLevel(logging.DEBUG)

        logfilepath = os.path.join(self.LOGROOT, logfilename)

        filefmt = "%(asctime)-15s %(levelname)s %(message)s"
        filedatefmt = "%a %d %b %Y %H:%M:%S"
        fileformatter = logging.Formatter(filefmt, filedatefmt)

        sfmt = "%(asctime)-15s %(levelname)s %(message)s"
        sdatefmt = "%d %b %H:%M"
        sformatter = logging.Formatter(sfmt, sdatefmt)

        fh = logging.FileHandler(logfilepath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fileformatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(sformatter)
        self.logger.addHandler(sh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == '__main__':
    log = Logger('NeuRec.log')
    log.debug('debug')
    log.info('info')
    log.warning('warning')
    log.error('error')
    log.critical('critical')

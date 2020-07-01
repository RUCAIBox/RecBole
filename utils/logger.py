import logging
import os


class Logger(object):

    def __init__(self, logfilename):
        """
        A logger that can show a message on standard output and write it into the
        file named `filename` simultaneously.
        All the message that you want to log MUST be str.

        :param logfilename: file name that log saved
        example:
            logger = Logger('train.log')
            logger.debug(train_state)
            logger.info(train_result)

        """

        self.LOGROOT='./log/'
        dir_name = os.path.dirname(self.LOGROOT)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(logfilename)
        self.logger.setLevel(logging.DEBUG)

        logfilepath = os.path.join(self.LOGROOT, logfilename)

        fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
        datefmt = "%a %d %b %Y %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)

        fh = logging.FileHandler(logfilepath)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
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

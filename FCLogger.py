# -*- coding: utf-8 -*-


# FuzzyClassificator - this program uses neural networks to solve classification problems,
# and uses fuzzy sets and fuzzy logic to interpreting results.
# Copyright (C) 2017, Timur Gilmullin
# e-mail: tim55667757@gmail.com


# This module implements universal logging system.


import sys
import logging


# initialize Main Parent Logger:
FCLogger = logging.getLogger("FCLogger")
formatString = "%(filename)-25s[Line:%(lineno)d]\t%(levelname)-10s[%(asctime)s]\t%(message)s"
formatter = logging.Formatter(formatString)
sys.stderr = sys.stdout


def SetLevel(vLevel='ERROR'):
    """
    This procedure setting up FCLogger verbosity level.
    """
    FCLogger.level = logging.NOTSET
    FCLogger.parent.level = logging.DEBUG  # fc.log always contains DEBUG log

    if isinstance(vLevel, str):
        if vLevel == '5' or vLevel.upper() == 'CRITICAL':
            FCLogger.level = logging.CRITICAL

        elif vLevel == '4' or vLevel.upper() == 'ERROR':
            FCLogger.level = logging.ERROR

        elif vLevel == '3' or vLevel.upper() == 'WARNING':
            FCLogger.level = logging.WARNING

        elif vLevel == '2' or vLevel.upper() == 'INFO':
            FCLogger.level = logging.INFO

        elif vLevel == '1' or vLevel.upper() == 'DEBUG':
            FCLogger.level = logging.DEBUG


class LevelFilter(logging.Filter):
    """
    Class using to set up log level filtering.
    """

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno >= self.level


def EnableLogger(logFile, parentHandler=FCLogger, useFormat=formatter):
    """
    Adding new file logger.
    """
    logHandler = logging.FileHandler(logFile)
    logHandler.level = logging.DEBUG
    logHandler.addFilter(LevelFilter(logging.DEBUG))

    if useFormat:
        logHandler.setFormatter(useFormat)

    else:
        logHandler.setFormatter(formatter)

    parentHandler.addHandler(logHandler)

    return logHandler


def DisableLogger(handler, parentHandler=FCLogger):
    """
    Disable given file logger.
    """
    if handler:
        handler.flush()
        handler.close()

    if handler in parentHandler.handlers:
        parentHandler.removeHandler(handler)


# Main init:

SetLevel('INFO')  # set up INFO verbosity level by default for FCLogger
streamHandler = logging.StreamHandler()  # initialize Console FCLogger by default
streamHandler.setFormatter(formatter)  # set formatter for console FCLogger

if FCLogger.parent.handlers:
    FCLogger.parent.handlers[0].setFormatter(formatter)

else:
    FCLogger.addHandler(streamHandler)  # adding console FCLogger handler to Parent Logger


# Constants and Global variables:

mainLogFile = 'fc.log'  # file to full logging by default
fileLogHandler = EnableLogger(logFile=mainLogFile, parentHandler=FCLogger, useFormat=formatter)  # add logging to file

sepLong = '-' * 80  # long log separator
sepShort = '-' * 40  # short log separator

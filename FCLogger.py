# -*- coding: utf-8 -*-


# FuzzyClassificator - this program uses neural networks to solve classification problems,
# and uses fuzzy sets and fuzzy logic to interpreting results.
# Copyright (C) 2014, Timur Gilmullin
# e-mail: tim55667757@gmail.com


# License: GNU GPL v3

# This file is part of FuzzyClassificator program.

# FuzzyClassificator is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# FuzzyClassificator program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with Foobar.
# If not, see <http://www.gnu.org/licenses/>.

# Этот файл - часть программы FuzzyClassificator.

# FuzzyClassificator - свободная программа: вы можете перераспространять ее и/или
# изменять ее на условиях Стандартной общественной лицензии GNU в том виде,
# в каком она была опубликована Фондом свободного программного обеспечения;
# либо версии 3 лицензии, либо (по вашему выбору) любой более поздней версии.

# Программа FuzzyClassificator распространяется в надежде, что она будет полезной,
# но БЕЗО ВСЯКИХ ГАРАНТИЙ; даже без неявной гарантии ТОВАРНОГО ВИДА
# или ПРИГОДНОСТИ ДЛЯ ОПРЕДЕЛЕННЫХ ЦЕЛЕЙ.
# Подробнее см. в Стандартной общественной лицензии GNU.

# Вы должны были получить копию Стандартной общественной лицензии GNU
# вместе с этой программой. Если это не так, см. <http://www.gnu.org/licenses/>.)


# This module implements universal logging system.


import sys
import logging


# initialize Main Parent Logger:
FCLogger = logging.getLogger("FCLogger")
formatString = "%(filename)-15s[Line:%(lineno)d]\t%(levelname)-10s[%(asctime)s]\t%(message)s"
formatter = logging.Formatter(formatString)
sys.stderr = sys.stdout


def SetLevel(vLevel=2):
    if vLevel >= 2:
        FCLogger.setLevel(logging.DEBUG) 

    elif vLevel == 1:
        FCLogger.setLevel(logging.INFO)

    elif vLevel <= 0:
        FCLogger.setLevel(logging.CRITICAL)

    else:
        FCLogger.setLevel(logging.DEBUG)


def EnableLogger(logFile, parentHandler=FCLogger, useFormat=formatter):
    logHandler = logging.FileHandler(logFile)

    if useFormat:
        logHandler.setFormatter(useFormat)

    else:
        logHandler.setFormatter(formatter)

    parentHandler.addHandler(logHandler)

    return logHandler


def DisableLogger(handler, parentHandler=FCLogger):
    if handler:
        handler.flush()
        handler.close()

    if handler in parentHandler.handlers:
        parentHandler.removeHandler(handler)


# Main init:

SetLevel(2)  # set up DEBUG verbosity level by default
streamHandler = logging.StreamHandler()  # initialize Console FCLogger by default
streamHandler.setFormatter(formatter)  # set formatter for console FCLogger
if FCLogger.parent.handlers:
    FCLogger.parent.handlers[0].setFormatter(formatter)
else:
    FCLogger.addHandler(streamHandler)  # adding console FCLogger handler to Parent Logger


# Constants and Global variables:

mainLogFile = 'fc.log'  # file to logging by default
fileLogHandler = EnableLogger(logFile=mainLogFile, parentHandler=FCLogger, useFormat=formatter)  # add logging to file

sepLong = '-' * 80  # long log separator
sepShort = '-' * 40  # short log separator

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


# Main runner. See help for usages. CLI-interaction supported.


import argparse

from PyBrainLearning import *
from FuzzyRoutines import *
from FCLogger import *


# Constants and Global variables:

ethalonsDataFile = 'ethalons.dat'  # file with ethalon data samples by default
candidatesDataFile = 'candidates.dat'  # file with candidates data samples by default
neuroNetworkFile = 'network.xml'  # file with Neuro Network configuration
reportDataFile = 'report.txt'  # Report file with classification analysis
bestNetworkFile = 'best_nn.xml'  # best network
bestNetworkInfoFile = 'best_nn.txt'  # information about best network

epochsToUpdate = 1  # epochs between error status updated

ignoreColumns = []  # List of ignored columns in input files.
ignoreRows = [1]  # List of ignored rows in input files.
sepSymbol = '\t'  # tab symbol used as separator by default

reloadNetworkFromFile = False  # reload or not Neuro Network from file before usage
noFuzzyOutput = False  # show results with fuzzy values if False, otherwise show real values


def ParseArgsMain():
    """
    Function get and parse command line keys.
    """
    parser = argparse.ArgumentParser()  # command-line string parser

    parser.description = 'This program uses neural networks to solve classification problems, and uses fuzzy sets and fuzzy logic to interpreting results.'
    parser.epilog = 'See examples on GitHub: https://github.com/Tim55667757/FuzzyClassificator FuzzyClassificator using Pyzo, http://www.pyzo.org - free and open-source computing environment, based on Python 3.3.2., and PyBrain library: http://pybrain.org'

    parser.add_argument('-l', '--debug-level', type=str, help='Use 1, 2, 3, 4, 5 or DEBUG, INFO, WARNING, ERROR, CRITICAL debug info verbosity, INFO (2) by default.')
    parser.add_argument('-e', '--ethalons', type=str, help='File with ethalon data samples, ethalons.dat by default.')
    parser.add_argument('-c', '--candidates', type=str, help='File with candidates data samples, candidates.dat by default.')
    parser.add_argument('-n', '--network', type=str, help='File with Neuro Network configuration, network.xml by default.')
    parser.add_argument('-r', '--report', type=str, help='Report file with classification analysis, report.txt by default.')
    parser.add_argument('-bn', '--best-network', type=str, help='Copy best network to this file, best_nn.xml by default.')
    parser.add_argument('-bni', '--best-network-info', type=str, help='File with information about best network, best_nn.txt by default.')

    parser.add_argument('-ic', '--ignore-col', type=str, help='Column indexes in input files that should be ignored. Use only dash and comma as separator numbers, other symbols are ignored. Example (no space after comma): 1,2,5-11')
    parser.add_argument('-ir', '--ignore-row', type=str, help='Row indexes in input files that should be ignored. Use only dash and comma as separator numbers, other symbols are ignored. 1st raw always set as ignored. Example (no space after comma): 2,4-7')
    parser.add_argument('-sep', '--separator', type=str, help='Separator symbol in raw data files. SPACE and TAB are reserved, TAB used by default.')
    parser.add_argument('--no-fuzzy', action='store_true', help='Do not show fuzzy results, only real. False by default.')
    parser.add_argument('--reload', action='store_true', help='Reload network from file before usage, False by default.')
    parser.add_argument('-u', '--epochs-to-update', type=int, help='Update error status after this epochs time, 1 by default. This parameter affected training speed.')

    parser.add_argument('--learn', type=str, nargs='+', help='Start program in learning mode with options (no space after comma): config=<inputs_num>,<layer1_num>,<layer2_num>,...,<outputs_num> epochs=<int_number> rate=<float_num> momentum=<float_num> epsilon=momentum=<float_num> stop=momentum=<float_num>')
    parser.add_argument('--classify', type=str, nargs='+', help='Start program in classificator mode with options (no space after comma): config=<inputs_num>,<layer1_num>,<layer2_num>,...,<outputs_num>')

    cmdArgs = parser.parse_args()
    if (cmdArgs.learn and cmdArgs.classify) or (not cmdArgs.learn and not cmdArgs.classify):
        parser.print_help()
        sys.exit()

    return cmdArgs


def LMStep1CreatingNetworkWithParameters(**kwargs):
    """
    This function realize Learning mode step:
    1. Creating PyBrain network instance with pre-defined config parameters.
    **kwargs are console parameters with user-define values.
    Function returns instance of PyBrain network.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 1. Creating PyBrain network instance with pre-defined config parameters.')

    # Create default config:
    config = ()  # network configuration
    epochs = 10  # epochs of learning
    rate = 0.05  # learning rate
    momentum = 0.01  # momentum of learning
    epsilon = 0.05  # epsilon used to compare the distance between the two vectors (may be with fuzzy values)
    stop = 5  # stop parameter

    # Updating config:
    if 'config' in kwargs.keys():
        try:
            # Parsing Neural Network Config parameter that looks like "config=inputs,layer1,layer2,...,outputs":
            config = tuple(int(par) for par in kwargs['config'].split(','))  # config for FuzzyNeuroNetwork

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Incorrect neural network config! Parameter config must looks like tuple of numbers: config=inputs,layer1,layer2,...,outputs')

    if 'epochs' in kwargs.keys():
        try:
            epochs = int(kwargs['epochs'])

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Epoch parameter might be an integer number greater or equal 1!')

    if 'rate' in kwargs.keys():
        try:
            rate = float(kwargs['rate'])

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Rate parameter might be a float number greater than 0 and less or equal 1!')

    if 'momentum' in kwargs.keys():
        try:
            momentum = float(kwargs['momentum'])

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Momentum parameter might be a float number greater than 0 and less or equal 1!')

    if 'epsilon' in kwargs.keys():
        try:
            epsilon = float(kwargs['epsilon'])

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Epsilon parameter might be a float number greater than 0 and less or equal 1!')

    if 'stop' in kwargs.keys():
        try:
            stop = float(kwargs['stop'])

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Stop parameter might be a float number greater than 0 and less or equal 100!')

    fNetwork = FuzzyNeuroNetwork()  # create network

    if noErrors:
        try:
            fNetwork.networkFile = neuroNetworkFile
            fNetwork.rawDataFile = ethalonsDataFile
            fNetwork.reportFile = reportDataFile
            fNetwork.bestNetworkFile = bestNetworkFile
            fNetwork.bestNetworkInfoFile = bestNetworkInfoFile
            fNetwork.config = config
            fNetwork.epochsToUpdate = epochsToUpdate

            if ignoreColumns:
                fNetwork.ignoreColumns = ignoreColumns  # set up ignored columns

            if ignoreRows:
                fNetwork.ignoreRows = ignoreRows  # set up ignored rows

            if sepSymbol:
                fNetwork.separator = sepSymbol  # set up separator symbol between columns in raw data files

            if epochs >= 0:
                fNetwork.epochs = epochs  # set up epochs of learning

            if rate:
                fNetwork.learningRate = rate  # set up learning rate parameter

            if momentum:
                fNetwork.momentum = momentum  # set up momentum of learning parameter

            if epsilon:
                fNetwork.epsilon = epsilon  # set up epsilon parameter

            if stop:
                fNetwork.stop = stop  # set up stop parameter

            FCLogger.debug('Instance of fuzzy network initialized with parameters:')
            FCLogger.debug('    {}'.format(kwargs))
            FCLogger.debug('File with ethalons data: {}'.format(os.path.abspath(fNetwork.rawDataFile)))
            FCLogger.debug('File for saving Neuronet: {}'.format(os.path.abspath(fNetwork.networkFile)))
            FCLogger.debug('Classification Report file: {}'.format(os.path.abspath(fNetwork.reportFile)))
            FCLogger.debug('For measurements used fuzzy scale:')

            for line in str(fNetwork.scale).split('\n'):
                FCLogger.debug(line)

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Failed to initialize the fuzzy network!')

    if noErrors:
        return fNetwork

    else:
        return None


def LMStep2ParsingRawDataFileWithEthalons(fNetwork):
    """
    This function realize Learning mode step:
    2. Parsing raw data file with ethalons.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns True if all operations with neural network finished successful.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 2. Parsing raw data file with ethalons.')

    fNetwork.ParseRawDataFile()

    if not fNetwork.rawData or not fNetwork.rawDefuzData:
        noErrors = False

    return noErrors


def LMStep3PreparingPyBrainDataset(fNetwork):
    """
    This function realize Learning mode step:
    3. Preparing PyBrain dataset.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns True if all operations with neural network finished successful.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 3. Preparing PyBrain dataset.')

    fNetwork.PrepareDataSet()

    if not fNetwork.dataSet:
        noErrors = False

    return noErrors


def LMStep4InitializePyBrainNetworkForLearning(fNetwork):
    """
    This function realize Learning mode step:
    4. Initialize empty PyBrain network for learning or reading network configuration from file.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns True if all operations with neural network finished successful.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 4. Initialize empty PyBrain network for learning or reading network configuration from file.')

    if reloadNetworkFromFile:
        fNetwork.LoadNetwork()  # reload old network for continuing training

    else:
        fNetwork.CreateNetwork()

    if not fNetwork.network:
        noErrors = False

    return noErrors


def LMStep5CreatingPyBrainTrainer(fNetwork):
    """
    This function realize Learning mode step:
    5. Creating PyBrain trainer.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns True if all operations with neural network finished successful.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 5. Creating PyBrain trainer.')

    fNetwork.CreateTrainer()

    if not fNetwork.trainer:
        noErrors = False

    return noErrors


def LMStep6StartsLearningAndSavingNetworkConfigurationToFile(fNetwork):
    """
    This function realize Learning mode step:
    6. Starts learning and saving network configuration to file.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns True if all operations with neural network finished successful.
    """
    FCLogger.info(sepShort)
    FCLogger.info('Step 6. Starts learning and saving network configuration to file.')

    noErrors = fNetwork.Train()  # train and receive finish status

    return noErrors


def LearningMode(**inputParameters):
    """
    Main function to work with input learn data and prepare neural network.
    Learning mode contain steps:
    1. Creating PyBrain network instance with pre-defined config parameters.
    2. Parsing raw data file with ethalons.
    3. Preparing PyBrain dataset.
    4. Initialize empty PyBrain network for learning or reading network configuration from file.
    5. Creating PyBrain trainer.
    6. Starts learning and saving network configuration to file.
    """
    successFinish = False  # success Learning Mode finish flag

    FCLogger.info(sepLong)
    FCLogger.info('Learning mode activated.')
    FCLogger.info('Log file: {}'.format(os.path.abspath(fileLogHandler.baseFilename)))

    fNetwork = LMStep1CreatingNetworkWithParameters(**inputParameters)

    if fNetwork:
        if LMStep2ParsingRawDataFileWithEthalons(fNetwork):
            if LMStep3PreparingPyBrainDataset(fNetwork):
                if LMStep4InitializePyBrainNetworkForLearning(fNetwork):
                    if LMStep5CreatingPyBrainTrainer(fNetwork):
                        if LMStep6StartsLearningAndSavingNetworkConfigurationToFile(fNetwork):
                            successFinish = True

    if successFinish:
        FCLogger.info(sepShort)
        FCLogger.info('Successful finish all Learning steps.')

        fNetwork.ClassificationResults(fullEval=True, needFuzzy=not(noFuzzyOutput))

    else:
        FCLogger.info(sepShort)
        FCLogger.critical('Learning finished with some errors!')

    FCLogger.info('Learning mode deactivated.')

    return successFinish


def CMStep1CreatingPyBrainNetwork(**kwargs):
    """
    This function realize Classifying mode step:
    1. Creating PyBrain network instance.
    Function returns instance of PyBrain network.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 1. Creating PyBrain network instance.')

    fNetwork = None
    config = ()  # network configuration

    if 'config' in kwargs.keys():
        try:
            # Parsing Neural Network Config parameter that looks like "config=inputs,layer1,layer2,...,outputs":
            config = tuple(int(par) for par in kwargs['config'].split(','))  # config for FuzzyNeuroNetwork

        except:
            noErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Incorrect neural network config! Parameter config must looks like tuple of numbers: config=inputs,layer1,layer2,...,outputs')

    try:
        fNetwork = FuzzyNeuroNetwork()  # create network

        fNetwork.config = config
        fNetwork.networkFile = neuroNetworkFile
        fNetwork.rawDataFile = candidatesDataFile
        fNetwork.reportFile = reportDataFile

        if ignoreColumns:
            fNetwork.ignoreColumns = ignoreColumns  # set up ignored columns

        if ignoreRows:
            fNetwork.ignoreRows = ignoreRows  # set up ignored rows

        if sepSymbol:
            fNetwork.separator = sepSymbol  # set up separator symbol between columns in raw data files

        FCLogger.debug('Instance of fuzzy network initialized with parameters:')
        FCLogger.debug('{}'.format(kwargs))
        FCLogger.debug('File with candidates data: {}'.format(os.path.abspath(fNetwork.rawDataFile)))
        FCLogger.debug('File for saving Neuronet: {}'.format(os.path.abspath(fNetwork.networkFile)))
        FCLogger.debug('Classification Report file: {}'.format(os.path.abspath(fNetwork.reportFile)))
        FCLogger.debug('For measurements used fuzzy scale:')

        for line in str(fNetwork.scale).split('\n'):
            FCLogger.debug(line)

    except:
        noErrors = False
        FCLogger.error(traceback.format_exc())
        FCLogger.error('Failed to initialize the fuzzy network!')

    if noErrors:
        return fNetwork

    else:
        return None


def CMStep2ParsingRawDataFileWithCandidates(fNetwork):
    """
    This function realize Classifying mode step:
    2. Parsing raw data file with candidates.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns True if all operations with neural network finished successful.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 2. Parsing raw data file with candidates.')

    fNetwork.ParseRawDataFile()

    if not fNetwork.rawData or not fNetwork.rawDefuzData:
        noErrors = False

    return noErrors


def CMStep3LoadingTrainedNetworkFromNetworkConfigurationFile(fNetwork):
    """
    This function realize Classifying mode step:
    3. Loading trained network from network configuration file.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns True if all operations with neural network finished successful.
    """
    noErrors = True  # successful flag
    FCLogger.info(sepShort)
    FCLogger.info('Step 3. Loading trained network from network configuration file.')

    fNetwork.LoadNetwork()  # reload old network for continuing training

    if not fNetwork.network:
        noErrors = False

    return noErrors


def CMStep4ActivatingNetworkForAllCandidateInputVectors(fNetwork):
    """
    This function realize Classifying mode step:
    4. Activating network for all candidate input vectors.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function returns result of classification.
    """
    FCLogger.info(sepShort)
    FCLogger.info('Step 4. Activating network for all candidate input vectors.')

    results = fNetwork.ClassificationResults(fullEval=True, needFuzzy=not(noFuzzyOutput), showExpectedVector=False)

    return results


def CMStep5InterpretingResults(fNetwork, results, fuzzyOutput=True):
    """
    This function realize Classifying mode step:
    5. Interpreting results.
    fNetwork is a PyBrain format neural network, created at 1st step.
    Function creates Classification Report File.
    """
    FCLogger.info(sepShort)
    FCLogger.info('Step 5. Interpreting results.')

    noErrors = fNetwork.CreateReport(results, fuzzyOutput)  # create report file

    return noErrors


def ClassifyingMode(**inputParameters):
    """
    Main function to work with input learn data and prepare neural network.
    Classifying mode contains steps:
    1. Creating PyBrain network instance.
    2. Parsing raw data file with candidates.
    3. Loading trained network from network configuration file.
    4. Activating network for all candidate input vectors.
    5. Interpreting results.
    """
    successFinish = False  # success Classifying mode finish flag

    FCLogger.info(sepLong)
    FCLogger.info('Classificator mode activated.')
    FCLogger.info('Log file: {}'.format(os.path.abspath(fileLogHandler.baseFilename)))

    fNetwork = CMStep1CreatingPyBrainNetwork(**inputParameters)

    if fNetwork:
        if CMStep2ParsingRawDataFileWithCandidates(fNetwork):
            if CMStep3LoadingTrainedNetworkFromNetworkConfigurationFile(fNetwork):
                classificateResult = CMStep4ActivatingNetworkForAllCandidateInputVectors(fNetwork)
                if classificateResult:
                    if CMStep5InterpretingResults(fNetwork, classificateResult, not(noFuzzyOutput)):
                        successFinish = True

    if successFinish:
        FCLogger.info(sepShort)
        FCLogger.info('Successful finish all Classifying steps.')

    else:
        FCLogger.info(sepShort)
        FCLogger.critical('Classifying finished with some errors!')

    FCLogger.info('Classificator mode deactivated.')

    return successFinish


if __name__ == "__main__":
    args = ParseArgsMain()  # get and parse command-line parameters
    exitCode = 0

    try:
        if args.debug_level:
            SetLevel(args.debug_level)  # set up FCLogger level

        if args.ethalons:
            ethalonsDataFile = args.ethalons  # raw data for learning

        if args.candidates:
            candidatesDataFile = args.candidates  # raw data for classify

        if args.network:
            neuroNetworkFile = args.network  # file with neural network configuration

        if args.report:
            reportDataFile = args.report  # report file with classification analysis

        if args.best_network:
            bestNetworkFile = args.best_network  # file with best network

        if args.best_network_info:
            bestNetworkInfoFile = args.best_network_info  # file with information about best network

        if args.ignore_col:
            ignoreColumns = DiapasonParser(args.ignore_col)

        if args.ignore_row:
            ignoreRows = DiapasonParser(args.ignore_row)

        if args.separator:
            sepSymbol = args.separator  # separator symbol: TAB, SPACE or another

        if args.no_fuzzy:
            noFuzzyOutput = True

        if args.reload:
            reloadNetworkFromFile = args.reload  # reload neural network from given file before usage

        if args.epochs_to_update:
            epochsToUpdate = args.epochs_to_update  # epochs before error status updating

        if args.learn:
            exitCode = int(not(LearningMode(**dict(kw.split('=') for kw in args.learn))))  # Learning mode

        elif args.classify:
            exitCode = int(not(ClassifyingMode(**dict(kw.split('=') for kw in args.classify))))  # Classifying mode

    except:
        exitCode = 1
        FCLogger.error(traceback.format_exc())

    finally:
        FCLogger.info('FuzzyClassificator work finished.')
        FCLogger.info(sepLong)

        DisableLogger(fileLogHandler)
        sys.exit(exitCode)
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


# Library for work with fuzzy neural networks.


import csv
import os
from datetime import datetime

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

from FuzzyRoutines import *
from FCLogger import FCLogger


class FuzzyNeuroNetwork(object):
    """
    Methods for work with raw-data and fuzzy neural networks.
    """

    def __init__(self):
        self.scale = UniversalFuzzyScale()  # creating fuzzy scale S_f = {Min, Low, Med, High, Max}
        self.networkFile = ''  # file with PyBrain network xml-configuration
        self.rawDataFile = ''  # file with text raw-data for learning
        self.config = ()  # network configuration is a tuple of numbers: (inputs_dim, layer1_dim,..., layerN_dim, outputs_dim)
        self._rawData = []  # list of raw strings data without first header line: ['input_vector',  'output_vector']
        self.headers = []  # list of strings with parsed headers
        self._rawDefuzData = []  # list of raw strings data but with deffazification values if it present in self._rawData
        self.dataSet = None  # PyBrain-formatted dataset after parsing raw-data: [[input_vector], [output_vector]]
        self.network = None  # PyBrain neural network instance
        self.trainer = None  # PyBrain trainer instance
        self._epochs = 10  # epochs of learning
        self._learningRate = 0.05  # learning rate
        self._momentum = 0.01  # momentum of learning
        self.reportFile = ''  # filename for report with classification analysis
        self._separator = '\t'  # tab symbol used as separator by default

    def _DefuzRawData(self):
        """
        Functions parse raw text data and converting fuzzy values to its real represents.
        """
        defuzData = []
        try:
            for line in self.rawData:
                # defuzzificating fuzzy values to real values:
                defuzzyValues = []

                for itemValue in line:
                    num = 0
                    try:
                        num = float(itemValue)

                    except:
                        level = self.scale.GetLevelByName(levelName=itemValue, exactMatching=False)
                        if level:
                            if isinstance(level['fSet'], FuzzySet):
                                num = level['fSet'].Defuz()

                        else:
                            FCLogger.warning('{} not correct real or fuzzy value! It is reset at 0.'.format(itemValue))

                    defuzzyValues.append(num)

                defuzData.append(defuzzyValues)

        except:
            defuzData = []
            FCLogger.error(traceback.format_exc())
            FCLogger.error('An error occurred while defuzzyficating values in raw data!')

        finally:
            self._rawDefuzData = defuzData

    @property
    def rawData(self):
        return self._rawData

    @rawData.setter
    def rawData(self, value):
        if isinstance(value, list):
            self._rawData = value

        else:
            self._rawData = []
            FCLogger.warning('Raw text data might be a list of strings! It was set to empty list: [].')

        self._DefuzRawData()

    @property
    def rawDefuzData(self):
        return self._rawDefuzData

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        if isinstance(value, int):

            if value >= 1:
                self._epochs = value

            else:
                self._epochs = 1
                FCLogger.warning('Parameter epochs might be greater or equal 1! It was set to 1 now.')

        else:
            self._epochs = 10
            FCLogger.warning('Parameter epochs might be an integer number! It was set to 10, by default.')

    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, value):
        if isinstance(value, float):

            if (value > 0) and (value <= 1):
                self._learningRate = value

            elif value <= 0:
                self._learningRate = 0.05
                FCLogger.warning('Parameter rate might be greater than 0! It was set to 0.05 now.')

            else:
                self._learningRate = 1
                FCLogger.warning('Parameter rate might be less than 1! It was set to 1 now.')

        else:
            self._epochs = 0.05
            FCLogger.warning('Parameter rate might be a float number! It was set to 0.05, by default.')

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, value):
        if isinstance(value, float):

            if (value > 0) and (value <= 1):
                self._momentum = value

            elif value <= 0:
                self._momentum = 0.01
                FCLogger.warning('Parameter momentum might be greater than 0! It was set to 0.01 now.')

            else:
                self._momentum = 1
                FCLogger.warning('Parameter momentum might be less than 1! It was set to 1 now.')

        else:
            self._momentum = 0.01
            FCLogger.warning('Parameter momentum might be a float number! It was set to 0.01, by default.')

    @property
    def separator(self):
        return self._separator

    @separator.setter
    def separator(self, value):
        if isinstance(value, str):

            if value.upper() == 'TAB':
                self._separator = '\t'

            elif value.upper() == 'SPACE':
                self._separator = ' '

            else:
                if len(value) == 1:
                    self._separator = value

                else:
                    FCLogger.warning('Parameter separator must be an 1-character string! It was set to TAB char, by default.')
                    self._separator = '\t'

        else:
            self._separator = '\t'
            FCLogger.warning('Parameter separator must be an 1-character string! It was set to TAB char, by default.')

    def ParseRawDataFile(self):
        """
        Get list of lines with raw string data without first header-line and empty lines.
        """
        raw = []
        FCLogger.info('Parsing file with raw data...')

        try:
            if self.rawDataFile:
                with open(self.rawDataFile, newline='') as csvfile:
                    FCLogger.debug('Opened file: {}'.format(self.rawDataFile))
                    FCLogger.debug('Separator symbol used: {}'.format('TAB' if self.separator == '\t' else '{}'.format('SPACE' if self.separator == ' ' else self.separator)))

                    for row in csv.reader(csvfile, delimiter=self._separator):
                        if row:
                            raw.append(row)

                if raw:
                    self.headers = raw[0]
                    raw = raw[1:]  # use data without first header-line

                    FCLogger.debug('Parsed raw-data vectors without first header-line:')
                    if len(raw) <= 5:
                        for line in raw:
                            FCLogger.debug('    {}'.format(line))

                    else:
                        FCLogger.debug('    {}'.format(raw[0]))
                        FCLogger.debug('    {}'.format(raw[1]))
                        FCLogger.debug('    [ ... skipped ... ]')
                        FCLogger.debug('    {}'.format(raw[-2]))
                        FCLogger.debug('    {}'.format(raw[-1]))

                FCLogger.info('File with raw data successfully parsed.')

            else:
                FCLogger.warning('File with raw data not define or not exist!')

        except:
            raw = []
            self.headers = []
            FCLogger.error(traceback.format_exc())
            FCLogger.error('An error occurred while parsing raw data file!')

        finally:
            if not raw:
                FCLogger.warning('Empty raw data file!')
            self.rawData = raw  # list of input vectors without first header line

    def PrepareDataSet(self):
        """
        This method preparing PyBrain dataset from raw data file.
        """
        learnData = None
        FCLogger.info('Converting parsed raw-data into PyBrain dataset format...')

        try:
            if self.config:

                if len(self.config) > 2:
                    learnData = SupervisedDataSet(self.config[0], self.config[-1])  # first and last values in config tuple

                else:
                    raise Exception('Network config must contains more than 2 parameters!')

            else:
                raise Exception('Network config not defined!')

            # add samples from defuz raw-data as [[input_vector], [output_vector]] for PyBrain network:
            for sample in self._rawDefuzData:
                learnData.addSample(sample[:self.config[0]], sample[self.config[0]:self.config[0] + self.config[-1]])

            FCLogger.debug('PyBrain dataset vectors, inputs and outputs (targets):')

            allInputs = learnData.data['input'][:learnData.endmarker['input']]
            learnDataInputsString = str(allInputs).split('\n')
            FCLogger.debug("- input vectors, dim({}, {}):".format(len(allInputs[0]), len(allInputs)))

            if len(allInputs) <= 10:
                for value in learnDataInputsString:
                    FCLogger.debug('    {}'.format(value))

            else:
                FCLogger.debug('    {}'.format(learnDataInputsString[0]))
                FCLogger.debug('    {}'.format(learnDataInputsString[1]))
                FCLogger.debug('     [... skipped ...]')
                FCLogger.debug('    {}'.format(learnDataInputsString[-2]))
                FCLogger.debug('    {}'.format(learnDataInputsString[-1]))

            allTargets = learnData.data['target'][:learnData.endmarker['target']]
            learnDataTargetsString = str(allTargets).split('\n')
            FCLogger.debug("- output vectors, dim({}, {}):".format(len(allTargets[0]), len(allTargets)))

            if len(allTargets) <= 10:
                for value in learnDataTargetsString:
                    FCLogger.debug('    {}'.format(value))

            else:
                FCLogger.debug('    {}'.format(learnDataTargetsString[0]))
                FCLogger.debug('    {}'.format(learnDataTargetsString[1]))
                FCLogger.debug('     [... skipped ...]')
                FCLogger.debug('    {}'.format(learnDataTargetsString[-2]))
                FCLogger.debug('    {}'.format(learnDataTargetsString[-1]))

            FCLogger.info('PyBrain dataset successfully prepared.')

        except:
            learnData = None
            FCLogger.error(traceback.format_exc())
            FCLogger.error('An error occurred while preparing PyBrain dataset!')

        finally:
            self.dataSet = learnData

    def CreateNetwork(self):
        """
        This method creating instance of PyBrain network.
        """
        net = None
        FCLogger.info('Creating PyBrain network...')

        try:
            if self.config:

                if len(self.config) > 2:
                    hLayers = self.config[1:-1]  # parameters for hidden layers

                    FCLogger.debug('Neuronet configuration: Config = <inputs, {layers}, outputs>')
                    FCLogger.debug('    - inputs is dimension of all input vectors: {},'.format(self.config[0]))
                    FCLogger.debug('    - outputs is dimension of all output vectors: {},'.format(self.config[-1]))
                    FCLogger.debug('    - count of hidden layers for Neuronet: {},'.format(len(hLayers)))

                    if len(hLayers) <= 10:
                        for nNum, dim in enumerate(hLayers):
                            FCLogger.debug('      ... dimension of {} hidden layer: {}'.format(nNum, dim))

                    else:
                        FCLogger.debug('      ... dimension of 0 hidden layer: {}'.format(hLayers[0]))
                        FCLogger.debug('      ... dimension of 1 hidden layer: {}'.format(hLayers[1]))
                        FCLogger.debug('      ... skipped ...')
                        FCLogger.debug('      ... dimension of {} hidden layer: {}'.format(len(hLayers) - 2, hLayers[-2]))
                        FCLogger.debug('      ... dimension of {} hidden layer: {}'.format(len(hLayers) - 1, hLayers[-1]))

                    net = buildNetwork(*self.config)  # create network with config

                else:
                    raise Exception('Network config must contains at least 3 parameters: (inputs_count, layer1_count, outputs_count)!')

            else:
                raise Exception('Network config not defined!')

            FCLogger.info('PyBrain network successfully created.')

        except:
            net = None
            FCLogger.error(traceback.format_exc())
            FCLogger.error('An error occurred while preparing PyBrain network!')

        finally:
            self.network = net

    def CreateTrainer(self):
        """
        This method preparing PyBrain trainer.
        """
        backpropTrainer = None
        FCLogger.info('Initializing PyBrain backpropagating trainer...')

        try:
            if self.network:

                if self.dataSet:
                    FCLogger.debug('Trainer using parameters:')
                    FCLogger.debug('    - PyBrain network previously created,')
                    FCLogger.debug('    - PyBrain dataset previously created,')
                    FCLogger.debug('    - epoch parameter: {}'.format(self._epochs))
                    FCLogger.debug('    - network learning rate parameter: {}'.format(self._learningRate))
                    FCLogger.debug('    - momentum parameter: {}'.format(self._momentum))

                    backpropTrainer = BackpropTrainer(self.network, self.dataSet, learningrate=self._learningRate, momentum=self._momentum)

                else:
                    raise Exception('PyBrain dataset not exist!')

            else:
                raise Exception('PyBrain network not exist!')

            FCLogger.info('PyBrain network successfully created.')

        except:
            backpropTrainer = None
            FCLogger.error(traceback.format_exc())
            FCLogger.error('An error occurred while creating PyBrain trainer!')

        finally:
            self.trainer = backpropTrainer

    def SaveNetwork(self):
        """
        Creating dump of network.
        """
        FCLogger.debug('Autosaving - enabled. Trying to save network as PyBrain xml-formatted file...')

        NetworkWriter.writeToFile(self.network, self.networkFile)

        FCLogger.info('{}Network saved to file: {}'.format('Current epoch = {}. '.format(self.trainer.epoch) if self.trainer.epoch else '',
                                                           os.path.abspath(self.networkFile)))

    def LoadNetwork(self):
        """
        Loading network dump from file.
        """
        FCLogger.debug('Loading network from PyBrain xml-formatted file...')
        net = None

        if os.path.exists(self.networkFile):
            net = NetworkReader.readFrom(self.networkFile)

            FCLogger.info('Network loaded from dump-file: {}'.format(os.path.abspath(self.networkFile)))

        else:
            FCLogger.warning('{} - file with Neural Network configuration not exist!'.format(os.path.abspath(self.networkFile)))

        self.network = net

    def ClassificationResultForOneVector(self, inputVector, expectedVector=None, needFuzzy=False):
        """
        Method use for receiving results after activating Neuronet with one input vector.
        inputVector - is a defuzzyficated raw data of input vector.
        if needFuzzy = True then appropriate output values converting into fuzzy values after activating, otherwise used real values.
        """
        # defuzzyficate input values:
        defuzInput = []
        for value in inputVector:
            level = self.scale.GetLevelByName(levelName='{}'.format(value), exactMatching=False)

            if level:
                if isinstance(level['fSet'], FuzzySet):
                    defuzInput.append(level['fSet'].Defuz())

            else:
                defuzInput.append(value)

        outputVector = self.network.activate(defuzInput)

        # defuzzyficate expected values:
        defuzExpectedVector = []
        if expectedVector:
            for value in expectedVector:
                level = self.scale.GetLevelByName(levelName='{}'.format(value), exactMatching=False)

                if level:
                    if isinstance(level['fSet'], FuzzySet):
                        defuzExpectedVector.append(level['fSet'].Defuz())

                else:
                    defuzExpectedVector.append(value)

            errorVector = []
            for num, currentValue in enumerate(outputVector):
                errorVector.append(float(defuzExpectedVector[num]) - currentValue)

        else:
            errorVector = None

        if needFuzzy:
            fuzzyOutputVector = []
            for value in outputVector:
                fuzzyOutputVector.append(self.scale.Fuzzy(value)['name'])

            if len(inputVector) <= 10:
                FCLogger.debug('        Input: {}\tOutput: {}{}'.format(inputVector, fuzzyOutputVector, '\tExpected: {}'.format(expectedVector) if expectedVector else ''))

            else:
                cutInput = '[{}, {}, ..., {}, {}]'.format(inputVector[0], inputVector[1], inputVector[-2], inputVector[-1])
                FCLogger.debug('        Input: {}\tOutput: {}{}'.format(cutInput, fuzzyOutputVector, '\tExpected: {}'.format(expectedVector) if expectedVector else ''))

            return inputVector, fuzzyOutputVector, expectedVector, errorVector

        else:
            if len(defuzInput) <= 10:
                FCLogger.debug('        Input: {}\tOutput: {}{}'.format(defuzInput, outputVector, '\tExpected: {}'.format(defuzExpectedVector) if expectedVector and defuzExpectedVector and errorVector else ''))

            else:
                cutDefuzInput = '[{}, {}, ..., {}, {}]'.format(defuzInput[0], defuzInput[1], defuzInput[-2], defuzInput[-1])
                FCLogger.debug('        Input: {}\tOutput: {}{}'.format(cutDefuzInput, outputVector, '\tExpected: {}'.format(defuzExpectedVector) if expectedVector and defuzExpectedVector and errorVector else ''))

            if expectedVector and defuzExpectedVector and errorVector:
                FCLogger.debug('        Error: {}'.format(errorVector))

            return defuzInput, outputVector, defuzExpectedVector, errorVector

    def ClassificationResults(self, fullEval=False, needFuzzy=False, showExpectedVector=True):
        """
        Method use for receiving results after activating Neuronet with all input vectors.
        If fullEval = True then method calculate results for all input vectors, otherwise for first and last two input vectors.
        If needFuzzy = True then appropriate output values converting into fuzzy values after activating, otherwise used real values.
        If showExpectedVector = True then vector with expected results will shown in log and result file.
        """
        classificationResults = []

        inputHeaders = self.headers[:self.config[0]]
        outputHeaders = self.headers[len(self.headers) - self.config[-1]:]

        if fullEval:
            FCLogger.debug('Full classification results:')

            if len(inputHeaders) <= 10:
                FCLogger.debug('    Header:    [{}]\t[{}]'.format(' '.join(head for head in inputHeaders),
                                                                  ' '.join(head for head in outputHeaders) if len(self.headers) >= self.config[0] + self.config[-1] else ''))

            else:
                FCLogger.debug('    Header:    [{} {} ... {} {}]\t[{}]'.format(inputHeaders[0], inputHeaders[1], inputHeaders[-2], inputHeaders[-1],
                                                                               ' '.join(head for head in outputHeaders) if len(self.headers) >= self.config[0] + self.config[-1] else ''))

            if needFuzzy:
                for vecNum, vector in enumerate(self._rawData):
                    inputVector = vector[:self.config[0]]

                    if showExpectedVector:
                        expectedVector = vector[len(vector) - self.config[-1]:]

                    else:
                        expectedVector = None

                    FCLogger.debug('    Vector #{}:'.format(vecNum))
                    classificationResults.append(self.ClassificationResultForOneVector(inputVector, expectedVector, needFuzzy))

            else:
                for vecNum, vector in enumerate(self._rawDefuzData):
                    inputVector = vector[:self.config[0]]

                    if showExpectedVector:
                        expectedVector = vector[len(vector) - self.config[-1]:]

                    else:
                        expectedVector = None

                    FCLogger.debug('    Vector #{}:'.format(vecNum))
                    classificationResults.append(self.ClassificationResultForOneVector(inputVector, expectedVector))

        else:
            FCLogger.debug('Some classification results:')

            if len(inputHeaders) <= 10:
                FCLogger.debug('    Header:    [{}]\t[{}]'.format(' '.join(head for head in inputHeaders),
                                                                  ' '.join(head for head in outputHeaders) if len(self.headers) >= self.config[0] + self.config[-1] else ''))

            else:
                FCLogger.debug('    Header:    [{} {} ... {} {}]\t[{}]'.format(inputHeaders[0], inputHeaders[1], inputHeaders[-2], inputHeaders[-1],
                                                                               ' '.join(head for head in outputHeaders) if len(self.headers) >= self.config[0] + self.config[-1] else ''))

            if len(self._rawData) <= 10:
                for vecNum, rawLine in enumerate(self._rawData):
                    FCLogger.debug('    Vector #{}:'.format(vecNum))
                    classificationResults.append(
                        self.ClassificationResultForOneVector(rawLine[:self.config[0]] if not needFuzzy else self._rawDefuzData[vecNum][:self.config[0]],
                                                              rawLine[len(rawLine) - self.config[-1]:] if not needFuzzy else self._rawDefuzData[vecNum][len(self._rawDefuzData[vecNum]) - self.config[-1]:], needFuzzy))

            else:
                FCLogger.debug('    Vector #0:')
                classificationResults.append(
                    self.ClassificationResultForOneVector(self._rawData[0][:self.config[0]] if not needFuzzy else self._rawDefuzData[0][:self.config[0]],
                                                          self._rawData[0][len(self._rawData[0]) - self.config[-1]:] if not needFuzzy else self._rawDefuzData[0][len(self._rawDefuzData[0]) - self.config[-1]:], needFuzzy))
                FCLogger.debug('    Vector #1:')
                classificationResults.append(
                    self.ClassificationResultForOneVector(self._rawData[1][:self.config[0]] if not needFuzzy else self._rawDefuzData[1][:self.config[0]],
                                                          self._rawData[1][len(self._rawData[1]) - self.config[-1]:] if not needFuzzy else self._rawDefuzData[1][len(self._rawDefuzData[1]) - self.config[-1]:], needFuzzy))
                FCLogger.debug('    ... skipped ...')
                FCLogger.debug('    Vector #{}:'.format(len(self._rawData) - 2))
                classificationResults.append(
                    self.ClassificationResultForOneVector(self._rawData[-2][:self.config[0]] if not needFuzzy else self._rawDefuzData[-2][:self.config[0]],
                                                          self._rawData[-2][len(self._rawData[-2]) - self.config[-1]:] if not needFuzzy else self._rawDefuzData[-2][len(self._rawDefuzData[-2]) - self.config[-1]:], needFuzzy))
                FCLogger.debug('    Vector #{}:'.format(len(self._rawData) - 1))
                classificationResults.append(
                    self.ClassificationResultForOneVector(self._rawData[-1][:self.config[0]] if not needFuzzy else self._rawDefuzData[-1][:self.config[0]],
                                                          self._rawData[-1][len(self._rawData[-1]) - self.config[-1]:] if not needFuzzy else self._rawDefuzData[-1][len(self._rawDefuzData[-1]) - self.config[-1]:], needFuzzy))

        return classificationResults

    def Train(self):
        """
        Realize training mechanism.
        """
        noTrainErrors = True  # successful train flag
        try:
            if self.trainer:
                started = datetime.now()
                FCLogger.debug('Max epochs: {}'.format(self._epochs))

                for epoch in range(self._epochs):
                    FCLogger.debug('Epoch: {}'.format(self.trainer.epoch + 1))

                    self.trainer.train()  # training network

                    self.ClassificationResults(fullEval=False, needFuzzy=False)  # show some results for ethalon vectors

                    if epoch % 10 == 0:
                        self.SaveNetwork()

                if self._epochs > 1:
                    self.SaveNetwork()
                FCLogger.info('Duration of learning: {}'.format(datetime.now() - started))

            else:
                raise Exception('Trainer instance not created!')

        except:
            noTrainErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('An error occurred while Training Fuzzy Network!')

        finally:
            return noTrainErrors

    def CreateReport(self, results=None, fuzzyOutput=True):
        """
        Creating text report after classificate vector-candidates.
        results is a list of tuples in ClassificationResults() format.
        fuzzyOutput is a key for show fuzzy values if True.
        """
        noReportCreationErrors = True  # successful of Creating Report process flag
        FCLogger.debug('Creating Classificate Report File...')

        try:
            if not results:
                results = self.ClassificationResults(fullEval=True, needFuzzy=fuzzyOutput)

            with open(self.reportFile, 'w') as fH:
                fH.write('Neuronet: {}\n\n'.format(os.path.abspath(self.networkFile)))
                fH.write('{}\n\n'.format(self.scale))
                fH.write('Classification results for candidates vectors:\n\n')
                head = '    Header: [{}]\t[{}]\n'.format(' '.join(header for header in self.headers[:self.config[0]]),
                                                         ' '.join(header for header in self.headers[len(self.headers) - self.config[-1]:]) if len(self.headers) >= self.config[0] + self.config[-1] else '')
                fH.write(head)
                fH.write('    {}\n'.format('-' * len(head) if len(head) < 100 else '-' * 100))

                for result in results:
                    if fuzzyOutput:
                        fH.write('    Input: {}\tOutput: {}{}\n'.format(
                            result[0], result[1], '\tExpected: {}'.format(result[2]) if result[2] else ''))

                    else:
                        fH.write('    Input: {}\tOutput: {}{}\n'.format(
                            result[0], result[1], '\tExpected: {}\tError: {}'.format(result[2], result[3]) if result[2] else ''))

            FCLogger.info('Classificate Report File created: {}'.format(os.path.abspath(self.reportFile)))

        except:
            noReportCreationErrors = False
            FCLogger.error(traceback.format_exc())
            FCLogger.error('An error occurred while Classificate Report creating!')

        finally:
            return noReportCreationErrors


if __name__ == "__main__":
    pass
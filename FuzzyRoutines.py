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
# If not, see <http:#www.gnu.org/licenses/>.

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


# Library contains some routines for work with fuzzy logic operators, fuzzy datasets and fuzzy scales.


import math
import copy
import traceback

from FCLogger import FCLogger


def DiapasonParser(diapason):
    """
    Parse input with diapason string and return list of full indexes in that diapason.
    Example: string "1,3,5-11" converts to list [1, 3, 5, 6, 7, 8, 9, 10 ,11]
    """
    fullDiapason = []

    try:
        for element in diapason.split(','):
            fullDiapason += [x for x in range(int(element.split('-')[0]), int(element.split('-')[-1]) + 1)]

    except:
        FCLogger.error('"{}" is not correct diapason string!'.format(diapason))
        fullDiapason = []

    finally:
        return sorted(list(set(fullDiapason)))


def IsCorrectFuzzyNumberValue(value):
    """
    All operations in fuzzy logic are executed with numbers in interval [0, 1].
    """
    correctNumberFlag = True

    try:
        correctNumberFlag = (0 <= value <= 1)

    except:
        FCLogger.error('{} is number not in [0, 1]!'.format(value))
        correctNumberFlag = False

    finally:
        return correctNumberFlag


def FuzzyNOT(fuzzyNumber, alpha=0.5):
    """
    Fuzzy logic NOT operator. y = 1 - Fuzzy if alpha = 0.5
    """
    result = fuzzyNumber  # return input number if errors

    if IsCorrectFuzzyNumberValue(fuzzyNumber) and IsCorrectFuzzyNumberValue(alpha) and alpha > 0:
        if (0 <= fuzzyNumber <= alpha):
            result = fuzzyNumber * (alpha - 1) / alpha + 1

        else:
            result = (fuzzyNumber - 1) * alpha / (alpha - 1)

    return result


def FuzzyNOTParabolic(fuzzyNumber, alpha=0.5, epsilon=0.001):
    """
    Parabolic fuzzy NOT operator. 2a - x - y = (2a - 1)(y - x)^2.
    """
    result = fuzzyNumber  # return input number if errors

    if IsCorrectFuzzyNumberValue(fuzzyNumber) and IsCorrectFuzzyNumberValue(alpha) and alpha > 0:
        if fuzzyNumber == 0:
            result = 1

        elif fuzzyNumber == 1:
            result = 0

        else:
            y = 0
            while (y <= 1) and abs((2 * alpha - fuzzyNumber - y) - (2 * alpha - 1) * (y - fuzzyNumber) ** 2) >= epsilon / 2:
                y += epsilon
            result = y

    return result


def FuzzyAND(aFuzzyNumber, bFuzzyNumber):
    """
    Fuzzy AND operator is minimum of two numbers.
    """
    result = 0  # return 0 if errors

    if IsCorrectFuzzyNumberValue(aFuzzyNumber) and IsCorrectFuzzyNumberValue(bFuzzyNumber):
        result = min(aFuzzyNumber, bFuzzyNumber)

    return result


def FuzzyOR(aFuzzyNumber, bFuzzyNumber):
    """
    Fuzzy OR operator is maximum of two numbers.
    """
    result = 1  # return 1 if errors

    if IsCorrectFuzzyNumberValue(aFuzzyNumber) and IsCorrectFuzzyNumberValue(bFuzzyNumber):
        result = max(aFuzzyNumber, bFuzzyNumber)

    return result


def TNorm(aFuzzyNumber, bFuzzyNumber, normType='logic'):
    """
    T-Norm conjunctive operators.
    normType is an operator's name:
        'logic' - result of fuzzy logic AND (min operator),
        'algebraic' - result of algebraic multiplication operation,
        'boundary' - result of boundary multiplication operation,
        'drastic' - result of drastic multiplication operation.
    """
    result = 0  # return 0 if errors

    if IsCorrectFuzzyNumberValue(aFuzzyNumber) and IsCorrectFuzzyNumberValue(bFuzzyNumber):
        if normType == 'logic':
            result = FuzzyAND(aFuzzyNumber, bFuzzyNumber)

        if normType == 'algebraic':
            result = aFuzzyNumber * bFuzzyNumber

        if normType == 'boundary':
            result = FuzzyOR(aFuzzyNumber + bFuzzyNumber - 1, 0)

        if normType == 'drastic':
            if aFuzzyNumber == 1:
                result = bFuzzyNumber

            elif bFuzzyNumber == 1:
                result = aFuzzyNumber

            else:
                result = 0

    return result


def TNormCompose(*fuzzyNumbers, normType='logic'):
    """
    T-Norm compose of n numbers.
    normType is an operator's name:
        'logic' - result of fuzzy logic AND (min operator),
        'algebraic' - result of algebraic multiplication operation,
        'boundary' - result of boundary multiplication operation,
        'drastic' - result of drastic multiplication operation.
    """
    result = 0  # return 0 if errors

    if len(fuzzyNumbers) >= 1:
        result = fuzzyNumbers[0]

        for f in fuzzyNumbers[1:]:
            result = TNorm(result, f, normType)

    return result


def SCoNorm(aFuzzyNumber, bFuzzyNumber, normType='logic'):
    """
    S-coNorm disjunctive operators.
    normType is an operator's name:
        'logic' - result of fuzzy logic OR (max operator),
        'algebraic' - result of algebraic addition operation,
        'boundary' - result of boundary addition operation,
        'drastic' - result of drastic addition operation.
    """
    result = 1  # return 1 if errors

    if IsCorrectFuzzyNumberValue(aFuzzyNumber) and IsCorrectFuzzyNumberValue(bFuzzyNumber):
        if normType == 'logic':
            result = FuzzyOR(aFuzzyNumber, bFuzzyNumber)

        if normType == 'algebraic':
            result = aFuzzyNumber + bFuzzyNumber - aFuzzyNumber * bFuzzyNumber

        if normType == 'boundary':
            result = FuzzyAND(aFuzzyNumber + bFuzzyNumber, 1)

        if normType == 'drastic':
            if aFuzzyNumber == 0:
                result = bFuzzyNumber

            elif bFuzzyNumber == 0:
                result = aFuzzyNumber

            else:
                result = 1

    return result


def SCoNormCompose(*fuzzyNumbers, normType='logic'):
    """
    S-coNorm compose of n numbers.
    normType is an operator's name:
        'logic' - result of fuzzy logic AND (min operator),
        'algebraic' - result of algebraic multiplication operation,
        'boundary' - result of boundary multiplication operation,
        'drastic' - result of drastic multiplication operation.
    """
    result = 0  # return 0 if errors

    if len(fuzzyNumbers) >= 1:
        result = fuzzyNumbers[0]

        for f in fuzzyNumbers[1:]:
            result = SCoNorm(result, f, normType)

    return result


class MFunction():
    """
    Routines for work with some default membership functions.
    """

    def __init__(self, userFunc, **membershipFunctionParams):
        self.accuracy = 1000  # Line of numbers divided by points, affect on accuracy, using in integral calculating
        self._functions = {'hyperbolic': self.Hyperbolic,
                           'bell': self.Bell,
                           'parabolic': self.Parabolic,
                           'triangle': self.Triangle,
                           'trapezium': self.Trapezium,
                           'exponential': self.Exponential,
                           'sigmoidal': self.Sigmoidal,
                           'desirability': self.Desirability}  # Factory registrator for all membership functions
        self.mju = self._functions[userFunc]  # Calculate result of define membership function

        if membershipFunctionParams or self.mju.__name__ == 'Desirability':
            self._parameters = membershipFunctionParams  # parameters for using in membership function

        else:
            raise Exception("You must specify all membership function's parameters!")

    @property
    def name(self):
        return self.mju.__name__  # membership function method name

    def __str__(self):
        # return view of function: Function_name(**parameters). Example: Bell(x, {"a": 0.6, "b": 0.66, "c": 0.77}
        funcView = '{}({})'.format(self.name, 'y' if self.name == 'Desirability' else 'x, {}'.format(
            '{' + ', '.join('"{}": {}'.format(*val) for val in [(k, self._parameters[k])
                                                                for k in sorted(self._parameters)]) + '}'))
        return funcView

    @property
    def parameters(self):
        return self._parameters  # all membership function parameters

    @parameters.setter
    def parameters(self, value):
        if value or self.mju.__name__ == 'Desirability':
            self._parameters = value

        else:
            raise Exception("You must specify all membership function's parameters!")

    def Hyperbolic(self, x):
        """
        This is hyperbolic membership function with real inputs x and parameters a, b, c.
        """
        a, b, c, result = 0, 0, 0, 0

        try:
            a = self._parameters['a']
            b = self._parameters['b']
            c = self._parameters['c']

            if x <= c:
                result = 1

            else:
                result = 1 / (1 + (a * (x - c)) ** b)

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Hyperbolic membership function use real inputs x and parameters a, b, c.')
            FCLogger.error('Your inputs: mju_hyperbolic({}, {}, {}, {})'.format(x, a, b, c))

        finally:
            return result

    def Bell(self, x):
        """
        This is bell membership function with real inputs x and parameters a, b, c.
        """
        a, b, c, result = 0, 0, 0, 0

        try:
            a = self._parameters['a']
            b = self._parameters['b']
            c = self._parameters['c']

            if x < b:
                result = self.Parabolic(x)

            elif (b <= x) and (x <= c):
                result = 1

            else:
                aOld = self._parameters['a']
                bOld = self._parameters['b']

                self._parameters['a'] = c
                self._parameters['b'] = c + b - a

                result = 1 - self.Parabolic(x)

                self._parameters['a'] = aOld
                self._parameters['b'] = bOld

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Bell membership function use real inputs x and parameters a, b, c.')
            FCLogger.error('Your inputs: mju_bell({}, {}, {}, {})'.format(x, a, b, c))

        finally:
            return result

    def Parabolic(self, x):
        """
        This is parabolic membership function with real inputs x and parameters a, b.
        """
        a, b, result = 0, 0, 0

        try:
            a = self._parameters['a']
            b = self._parameters['b']

            if x <= a:
                result = 0

            elif (a < x) and (x <= (a + b) / 2):
                result = (2 * (x - a) ** 2) / (b - a) ** 2

            elif ((a + b) / 2 < x) and (x < b):
                result = 1 - (2 * (x - b) ** 2) / (b - a) ** 2

            else:
                result = 1

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Parabolic membership function use real inputs x and parameters a, b.')
            FCLogger.error('Your inputs: mju_parabolic({}, {}, {})'.format(x, a, b))

        finally:
            return result

    def Triangle(self, x):
        """
        This is triangle membership function with real inputs x and parameters a, b, c.
        """
        a, b, c, result = 0, 0, 0, 0

        try:
            a = self._parameters['a']
            b = self._parameters['b']
            c = self._parameters['c']

            if x <= a:
                result = 0

            elif (a < x) and (x <= c):
                result = (x - a) / (c - a)

            elif (c < x) and (x < b):
                result = (b - x) / (b - c)

            else:
                result = 0

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Triangle membership function use real inputs x and parameters a, b, c.')
            FCLogger.error('Your inputs: mju_triangle({}, {}, {}, {})'.format(x, a, b, c))

        finally:
            return result

    def Trapezium(self, x):
        """
        This is trapezium membership function with real inputs x and parameters a, b, c, d.
        """
        a, b, c, d, result = 0, 0, 0, 0, 0

        try:
            a = self._parameters['a']
            b = self._parameters['b']
            c = self._parameters['c']
            d = self._parameters['d']

            if x < a:
                result = 0

            elif (a < x) and (x < c):
                result = (x - a) / (c - a)

            elif (c <= x) and (x <= d):
                result = 1

            elif (d < x) and (x <= b):
                result = (b - x) / (b - d)

            else:
                result = 0

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Trapezium membership function use real inputs x and parameters a, b, c, d.')
            FCLogger.error('Your inputs: mju_trapezium({}, {}, {}, {}, {})'.format(x, a, b, c, d))

        finally:
            return result

    def Exponential(self, x):
        """
        This is exponential membership function with real inputs x and parameters a, b.
        """
        a, b, result = 0, 0, 0

        try:
            a = self._parameters['a']
            b = self._parameters['b']

            if b != 0:
                result = math.exp(1) ** (-0.5 * ((x - a) / b) ** 2)

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Exponential membership function use real inputs x and parameters a, b.')
            FCLogger.error('Your inputs: mju_exponential({}, {}, {})'.format(x, a, b))

        finally:
            return result

    def Sigmoidal(self, x):
        """
        This is sigmoidal membership function with real inputs x and parameters a, b.
        """
        a, b, result = 0, 0, 0

        try:
            a = self._parameters['a']
            b = self._parameters['b']

            result = 1 / (1 + math.exp(1) ** (-a * (x - b)))

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error('Sigmoidal membership function use real inputs x and parameters a, b.')
            FCLogger.error('Your inputs: mju_sigmoidal({}, {}, {})'.format(x, a, b))

        finally:
            return result

    def Desirability(self, y):
        """
        This is Harrington's desirability membership function with real input y without any parameters.
        """
        result = 0

        try:
            result = math.exp(-math.exp(-y))

        except:
            result = 0
            FCLogger.error(traceback.format_exc())
            FCLogger.error("Harrington's desirability membership function use only real input y without any parameters.")
            FCLogger.error('Your inputs: mju_desirability({})'.format(y))

        finally:
            return result


class FuzzySet():
    """
    Routines for work with fuzzy sets.
    Fuzzy set A = <membershipFunction, supportSet>
    """

    def __init__(self, membershipFunction, supportSet=(0., 1.), linguisticName='FuzzySet'):
        if isinstance(linguisticName, str):
            self._name = linguisticName

        else:
            raise Exception("Linguistic name of Fuzzy Set must be a string value!")

        if isinstance(membershipFunction, MFunction):
            self._mFunction = membershipFunction  # instance of MembershipFunction class

        else:
            raise Exception('Not MFunction class instance was given!')

        if isinstance(supportSet, tuple) and (len(supportSet) == 2) and (supportSet[0] < supportSet[1]):
            self._supportSet = supportSet  # support set of given membership function

        else:
            raise Exception('Support Set must be 2-dim tuple (a, b) with real a, b parameters, a < b!')

    def __str__(self):
        # return view of fuzzy set - name = <mju(x|y, params), supportSet>. Example: FuzzySet = <Bell(x, a, b), [0, 1]>
        fSetView = '{} = <{}, [{}, {}]>'.format(self._name, self._mFunction, self._supportSet[0], self._supportSet[1])
        return fSetView

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value

        else:
            raise Exception("Linguistic name of Fuzzy Set must be a string value!")

    @property
    def mFunction(self):
        return self._mFunction  # current membership

    @mFunction.setter
    def mFunction(self, value):
        if isinstance(value, MFunction):
            self._mFunction = value

        else:
            raise Exception('Not MFunction class instance was given!')

    @property
    def supportSet(self):
        return self._supportSet

    @supportSet.setter
    def supportSet(self, value):
        if isinstance(value, tuple) and (len(value) == 2) and (value[0] < value[1]):
            self._supportSet = value  # new support set of given membership function

        else:
            raise Exception('Support Set must be 2-dim tuple (a, b) with real a, b parameters, a < b!')

    def Defuz(self):
        """
        Defuzzyfication function returns real value in support set of given fuzzy set using "center of gravity method".
        Integrals in this method calculated from left to right border of support set of membership function.
        Integrals are approximately calculated by Newton-Leibniz formula.
        """
        left = self._supportSet[0]
        right = self._supportSet[1]
        step = (right - left) / self._mFunction.accuracy

        numeratorIntegral = 0
        denominatorIntegral = 0

        for iteration in range(self._mFunction.accuracy):
            x = left + (iteration + 1) * step
            mjuValue = self._mFunction.mju(x)

            numeratorIntegral += x * mjuValue
            denominatorIntegral += mjuValue

        return numeratorIntegral / denominatorIntegral


class FuzzyScale():
    """
    Routines for work with fuzzy scales. Fuzzy scale is an ordered set of linguistic variables.
    Fuzzy scale contains named levels and its MF. This object looks like this:
    S = [{'name': 'name_1', 'fSet': fuzzySet_1},
         {'name': 'name_2', 'fSet': fuzzySet_2}, ...]
        where name-key is a linguistic name of fuzzy set,
        fSet-key is a user define fuzzy set, an instance of FuzzySet class.
    """

    def __init__(self):
        self._name = 'DefaultScale'  # default scale contains 3 level, DefaultScale = {Min, Med, High}:

        self._levels = [{'name': 'Min',
                         'fSet': FuzzySet(membershipFunction=MFunction('hyperbolic', **{'a': 7, 'b': 4, 'c': 0}),
                                          supportSet=(0., 1.),
                                          linguisticName='Minimum')},
                        {'name': 'Med',
                         'fSet': FuzzySet(membershipFunction=MFunction('bell', **{'a': 0.35, 'b': 0.5, 'c': 0.6}),
                                          supportSet=(0., 1.),
                                          linguisticName='Medium')},
                        {'name': 'High',
                         'fSet': FuzzySet(membershipFunction=MFunction('triangle', **{'a': 0.7, 'b': 1, 'c': 1}),
                                          supportSet=(0., 1.),
                                          linguisticName='High')}]

    def __str__(self):
        # return view of fuzzy scale - name = {**levels} and levels interpreter. Example:
        # DefaultScale = {Min, Med, High}
        #     Minimum = <Hyperbolic(x, {"a": 7, "b": 4, "c": 0}), [0.0, 1.0]>
        #     Medium = <Bell(x, {"a": 0.35, "b": 0.5, "c": 0.6}), [0.0, 1.0]>
        #     High = <Triangle(x, {"a": 0.7, "b": 1, "c": 1}), [0.0, 1.0]>
        allLevelsName = self._levels[0]['name']
        allLevels = '\n    {}'.format(self._levels[0]['fSet'].__str__())
        
        for level in self._levels[1:]:
            allLevelsName += ', {}'.format(level['name'])
            allLevels += '\n    {}'.format(str(level['fSet']))
        
        scaleView = '{} = {{{}}}{}'.format(self._name, allLevelsName, allLevels)
        
        return scaleView

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value

        else:
            raise Exception("Name of Fuzzy Scale must be a string value!")

    @property
    def levels(self):
        return self._levels

    @levels.setter
    def levels(self, value):
        if value:
            for level in value:
                if isinstance(level, dict) and (len(level) == 2) and ('name' and 'fSet' in level.keys()):
                    if not isinstance(level['name'], str):
                        raise Exception("Level name - 'name' parameter - must be a string value!")

                    if not isinstance(level['fSet'], FuzzySet):
                        raise Exception("Fuzzy set - 'fSet' parameter - must be an instance of FuzzySet class!")

                    nameCount = 0  # check for unique name:
                    for otherLevel in value:
                        if otherLevel['name'] == level['name']:
                            nameCount += 1

                    if nameCount > 1:
                        raise Exception("The scale contains no unique levels! Warning for: {}".format(level['name']))

                else:
                    raise Exception("Level of fuzzy scale must be 2-dim dictionary looks like {'name': 'level_name', 'fSet': FuzzySet_instance}!")

            self._levels = value

        else:
            raise Exception('Fuzzy scale must contain at least one linguistic variable!')

    def Fuzzy(self, realValue):
        """
        Fuzzyfication function returns one of levels on fuzzy scale for given real value who MF(value) are highest.
        """
        fuzzyLevel = self._levels[0]

        for level in self._levels[1:]:
            if fuzzyLevel['fSet'].mFunction.mju(realValue) <= level['fSet'].mFunction.mju(realValue):
                fuzzyLevel = level
                
        return fuzzyLevel

    def GetLevelByName(self, levelName, exactMatching=True):
        """
        Function return fuzzy level as dictionary level = {'name': 'level_name', 'fSet': fuzzySet}
        exactMatching is a flag for exact matching search,
            if True then levelName must be equal to level['name'],
            otherwise - level['name'] in uppercase must contains levelName in uppercase.
        """
        fuzzyLevel = None

        for level in self._levels:
            scaleLevelName = level['name']

            if not exactMatching:
                levelName = levelName.upper()
                scaleLevelName = scaleLevelName.upper()

                if levelName in scaleLevelName:
                    fuzzyLevel = level

            else:
                if levelName == scaleLevelName:
                    fuzzyLevel = level

            if fuzzyLevel:
                break

        return fuzzyLevel


class UniversalFuzzyScale(FuzzyScale):
    """
    Iniversal fuzzy scale S_f = {Min, Low, Med, High, Max}. Example view:
    FuzzyScale = {Min, Low, Med, High, Max}
        Min = <Hyperbolic(x, {"a": 8, "b": 20, "c": 0}), [0.0, 0.23]>
        Low = <Bell(x, {"a": 0.17, "b": 0.23, "c": 0.34}), [0.17, 0.4]>
        Med = <Bell(x, {"a": 0.34, "b": 0.4, "c": 0.6}), [0.34, 0.66]>
        High = <Bell(x, {"a": 0.6, "b": 0.66, "c": 0.77}), [0.6, 0.83]>
        Max = <Parabolic(x, {"a": 0.77, "b": 0.95}), [0.77, 1.0]>
    """

    def __init__(self):
        super().__init__()

        self._name = 'FuzzyScale'  # default scale contains 3 level, FuzzyScale = {Min, Med, High}:

        self._levels = [{'name': 'Min',
                         'fSet': FuzzySet(membershipFunction=MFunction('hyperbolic', **{'a': 8, 'b': 20, 'c': 0}),
                                          supportSet=(0., 0.23),
                                          linguisticName='Min')},
                        {'name': 'Low',
                         'fSet': FuzzySet(membershipFunction=MFunction('bell', **{'a': 0.17, 'b': 0.23, 'c': 0.34}),
                                          supportSet=(0.17, 0.4),
                                          linguisticName='Low')},
                        {'name': 'Med',
                         'fSet': FuzzySet(membershipFunction=MFunction('bell', **{'a': 0.34, 'b': 0.4, 'c': 0.6}),
                                          supportSet=(0.34, 0.66),
                                          linguisticName='Med')},
                        {'name': 'High',
                         'fSet': FuzzySet(membershipFunction=MFunction('bell', **{'a': 0.6, 'b': 0.66, 'c': 0.77}),
                                          supportSet=(0.6, 0.83),
                                          linguisticName='High')},
                        {'name': 'Max',
                         'fSet': FuzzySet(membershipFunction=MFunction('parabolic', **{'a': 0.77, 'b': 0.95}),
                                          supportSet=(0.77, 1.),
                                          linguisticName='Max')}]

    @property
    def levels(self):
        return self._levels  # only read levels for Universal Fuzzy Scale


if __name__ == "__main__":
    ## Some examples (just run this FuzzyRoutines module):

    ## --- Usage of some membership functions (uncomment one of them):

    #mjuPars = {'a': 7, 'b': 4, 'c': 0}  # hyperbolic params example
    #funct = MFunction(userFunc='hyperbolic', **mjuPars)  # creating instance of hyperbolic function

    #mjuPars = {'a': 0, 'b': 0.3, 'c': 0.4}  # bell params example
    #funct = MFunction(userFunc='bell', **mjuPars)  # creating instance of bell function

    #mjuPars = {'a': 0, 'b': 1}  # parabolic params example
    #funct = MFunction(userFunc='parabolic', **mjuPars)  # creating instance of parabolic function

    #mjuPars = {'a': 0.2, 'b': 0.8, 'c': 0.7}  # triangle params example
    #funct = MFunction(userFunc='triangle', **mjuPars)  # creating instance of triangle function

    mjuPars = {'a': 0.1, 'b': 1, 'c': 0.5, 'd': 0.8}  # trapezium params example
    funct = MFunction(userFunc='trapezium', **mjuPars)  # creating instance of trapezium function

    #mjuPars = {'a': 0.5, 'b': 0.15}  # exponential params example
    #funct = MFunction(userFunc='exponential', **mjuPars)  # creating instance of exponential function

    #mjuPars = {'a': 15, 'b': 0.5}  # sigmoidal params example
    #funct = MFunction(userFunc='sigmoidal', **mjuPars)  # creating instance of sigmoidal function

    #funct = MFunction(userFunc='desirability')  # creating instance of desirability function without parameters

    print('Printing Membership function with parameters: ', funct)

    ## --- Calculating some function's values in [0, 1]:

    xPar = 0
    for i in range(0, 11, 1):
        xPar = (xPar + i) / 10
        res = funct.mju(xPar)  # calculate one value of MF with given parameters
        print('{} = {:1.4f}'.format(funct, res))

    ## --- Work with fuzzy set:

    fuzzySet = FuzzySet(funct, (0., 1.))  # creating fuzzy set A = <mju_funct, support_set>
    print('Printing fuzzy set after init and before changes:', fuzzySet)
    print('Defuz({}) = {:1.2f}'.format(fuzzySet.name, fuzzySet.Defuz()))

    changedMjuPars = copy.deepcopy(mjuPars)  # change parameters of membership function with deepcopy example:
    changedMjuPars['a'] = 0
    changedMjuPars['b'] = 1
    changedSupportSet = (0.5, 1)  # change support set
    fuzzySet.name = 'Changed fuzzy set'

    fuzzySet.mFunction.parameters = changedMjuPars
    fuzzySet.supportSet = changedSupportSet

    print('New membership function with parameters: ', fuzzySet.mFunction)
    print('New support set: ', fuzzySet.supportSet)
    print('New value of Defuz({}) = {:1.2f}'.format(fuzzySet.name, fuzzySet.Defuz()))
    print('Printing fuzzy set after changes:', fuzzySet)

    ## --- Work with fuzzy scales:
    ## Fuzzy scale is an ordered set of linguistic variables that looks like this:
    ## S = [{'name': 'name_1', 'fSet': fuzzySet_1}, {'name': 'name_2', 'fSet': fuzzySet_2}, ...],
    ##     where name is a linguistic name of fuzzy set,
    ##     fSet is a user define fuzzy set of FuzzySet type.
    scale = FuzzyScale()  # intialize new fuzzy scale with default levels

    print('Printing default fuzzy scale in human-readable:', scale)

    print('Defuz() of all default levels:')
    for item in scale.levels:
        print('Defuz({}) = {:1.2f}'.format(item['name'], item['fSet'].Defuz()))

    print('Define some new levels:')

    minFunct = MFunction('hyperbolic', **{'a': 2, 'b': 20, 'c': 0})
    levelMin = FuzzySet(membershipFunction=minFunct, supportSet=(0., 0.5), linguisticName='min')
    print('Printing Level 1 in human-readable:', levelMin)

    medFunct = MFunction('bell', **{'a': 0.4, 'b': 0.55, 'c': 0.7})
    levelMed = FuzzySet(membershipFunction=medFunct, supportSet=(0.25, 0.75), linguisticName='med')
    print('Printing Level 2 in human-readable:', levelMed)

    maxFunct = MFunction('triangle', **{'a': 0.65, 'b': 1, 'c': 1})
    levelMax = FuzzySet(membershipFunction=maxFunct, supportSet=(0.7, 1.), linguisticName='max')
    print('Printing Level 3 in human-readable:', levelMax)

    scale.name = 'New Scale'
    scale.levels = [{'name': levelMin.name, 'fSet': levelMin},
                    {'name': levelMed.name, 'fSet': levelMed},
                    {'name': levelMax.name, 'fSet': levelMax}]  # add new ordered set of linguistic variables into scale

    print('Changed List of levels as objects:', scale.levels)
    print('Printing changed fuzzy scale in human-readable:', scale)

    print('Defuz() of all New Scale levels:')
    for item in scale.levels:
        print('Defuz({}) = {:1.2f}'.format(item['name'], item['fSet'].Defuz()))

    ## --- Work with Universal Fuzzy Scale:
    ## Iniversal fuzzy scales S_f = {Min, Low, Med, High, Max} pre-defined in UniversalFuzzyScale() class.

    uniFScale = UniversalFuzzyScale()
    print('Levels of Universal Fuzzy Scale:', uniFScale.levels)
    print('Printing scale:', uniFScale)

    ## Use Fuzzy() function to looking for level on Fuzzy Scale:

    xPar = 0
    for i in range(0, 10, 1):
        xPar = (xPar + i) / 10
        res = uniFScale.Fuzzy(xPar)  # calculate fuzzy level for some real values
        print('Fuzzy({:1.1f}, {}) = {}, {}'.format(xPar, uniFScale.name, res['name'], res['fSet']))

    ## Finding fuzzy level using GetLevelByName() function:

    print('Finding level by name with exact matching:')

    res = uniFScale.GetLevelByName('Min')
    print('GetLevelByName(Min, {}) = {}, {}'.format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('High')
    print('GetLevelByName(High, {}) = {}, {}'.format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('max')
    print('GetLevelByName(max, {}) = {}, {}'.format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    print('Finding level by name without exact matching:')

    res = uniFScale.GetLevelByName('mIn', exactMatching=False)
    print("GetLevelByName('mIn', {}) = {}, {}".format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('max', exactMatching=False)
    print("GetLevelByName('max', {}) = {}, {}".format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('Hig', exactMatching=False)
    print("GetLevelByName('Hig', {}) = {}, {}".format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('LOw', exactMatching=False)
    print("GetLevelByName('LOw', {}) = {}, {}".format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('eD', exactMatching=False)
    print("GetLevelByName('eD', {}) = {}, {}".format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('Highest', exactMatching=False)
    print("GetLevelByName('Highest', {}) = {}, {}".format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    ## --- Work with fuzzy logic operators:

    ## Checks that number is in [0, 1]:

    print('IsCorrectFuzzyNumberValue(0.5) =', IsCorrectFuzzyNumberValue(0.5))
    print('IsCorrectFuzzyNumberValue(1.1) =', IsCorrectFuzzyNumberValue(1.1))

    ## Calculates result of fuzzy NOT, fuzzy NOT with alpha parameter and parabolic fuzzy NOT operations:

    print('FNOT(0.25) =', FuzzyNOT(0.25))
    print('FNOT(0.25, alpha=0.25) =', FuzzyNOT(0.25, alpha=0.25))
    print('FNOT(0.25, alpha=0.75) =', FuzzyNOT(0.25, alpha=0.75))
    print('FNOT(0.25, alpha=1) =', FuzzyNOT(0.25, alpha=1))

    print('FNOTParabolic(0.25, alpha=0.25) =', FuzzyNOTParabolic(0.25, alpha=0.25))
    print('FNOTParabolic(0.25, alpha=0.75) =', FuzzyNOTParabolic(0.25, alpha=0.75))

    ## Calculates result of fuzzy AND/OR operations:

    print('FuzzyAND(0.25, 0.5) =', FuzzyAND(0.25, 0.5))
    print('FuzzyOR(0.25, 0.5) =', FuzzyOR(0.25, 0.5))

    ## Calculates result of T-Norm operations, where T-Norm is one of conjunctive operators - logic, algebraic, boundary, drastic:

    print("TNorm(0.25, 0.5, 'logic') =", TNorm(0.25, 0.5, normType='logic'))
    print("TNorm(0.25, 0.5, 'algebraic') =", TNorm(0.25, 0.5, normType='algebraic'))
    print("TNorm(0.25, 0.5, 'boundary') =", TNorm(0.25, 0.5, normType='boundary'))
    print("TNorm(0.25, 0.5, 'drastic') =", TNorm(0.25, 0.5, normType='drastic'))

    ## Calculates result of S-coNorm operations, where S-coNorm is one of disjunctive operators - logic, algebraic, boundary, drastic:

    print("SCoNorm(0.25, 0.5, 'logic') =", SCoNorm(0.25, 0.5, normType='logic'))
    print("SCoNorm(0.25, 0.5, 'algebraic') =", SCoNorm(0.25, 0.5, normType='algebraic'))
    print("SCoNorm(0.25, 0.5, 'boundary') =", SCoNorm(0.25, 0.5, normType='boundary'))
    print("SCoNorm(0.25, 0.5, 'drastic') =", SCoNorm(0.25, 0.5, normType='drastic'))

    ## Calculates result of T-Norm operations for N numbers, N > 2:

    print("TNormCompose(0.25, 0.5, 0.75, 'logic') =", TNormCompose(0.25, 0.5, 0.75, normType='logic'))
    print("TNormCompose(0.25, 0.5, 0.75, 'algebraic') =", TNormCompose(0.25, 0.5, 0.75, normType='algebraic'))
    print("TNormCompose(0.25, 0.5, 0.75, 'boundary') =", TNormCompose(0.25, 0.5, 0.75, normType='boundary'))
    print("TNormCompose(0.25, 0.5, 0.75, 'drastic') =", TNormCompose(0.25, 0.5, 0.75, normType='drastic'))

    ## Calculates result of S-coNorm operations for N numbers, N > 2:

    print("SCoNormCompose(0.25, 0.5, 0.75, 'logic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='logic'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'algebraic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='algebraic'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'boundary') =", SCoNormCompose(0.25, 0.5, 0.75, normType='boundary'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'drastic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='drastic'))
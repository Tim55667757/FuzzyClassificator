FuzzyClassificator
==================

This program uses neural networks to solve classification problems, and uses fuzzy sets and fuzzy logic to interpreting results. FuzzyClassificator provided under a license GNU GPL v3.


How to use
--------------

FuzzyClassificator uses ethalons.dat (default) as learning data and candidates.dat (default) for classifying data (See "Preparing data" chapter).
Work contains two steps:

1. Learning. At this step program parses ethalon data, learning neural network on this data and then saves neural network configuration into file.

2. Classifying. At this step program uses trained network for classification candidates from data file.


**Presets:**

FuzzyClassificator using Pyzo, http://www.pyzo.org - free and open-source computing environment, based on Python 3.3.2 and includes many scientific packages, e.g. PyBrain library, http://pybrain.org - neural network routines.


**Usage:**

    python FuzzyClassificator.py [options] [--learn]|[--classify] [Network_Options]


*Optional arguments:*

    -h, --help
        Show help message and exit.

    -l [verbosity], --debug-level=[verbosity]
        Use 1, 2, 3, 4, 5 or DEBUG, INFO, WARNING, ERROR, CRITICAL debug info verbosity,
        INFO (2) by default.

    -e [ethalon_filename], --ethalons=[ethalon_filename]
        File with ethalon data samples, ethalons.dat by default.

    -c [candidates_filename], --candidates=[candidates_filename]
        File with candidates data samples, candidates.dat by default.

    -n [network_filename], --network=[network_filename]
        File with Neuro Network configuration, network.xml by default.

    -r [report_filename], --report=[report_filename]
        File with Neuro Network configuration, report.txt by default.

    -bn [best_network_filename], --best-network=[best_network_filename]
        Copy best network to this file, best_nn.xml by default.

    -bni [best_network_info_filename], --best-network-info=[best_network_info_filename]
        File with information about best network, best_nn.txt by default.

    -ic [indexes], --ignore-col=[indexes]
        Columns in input files that should be ignored.
        Use only dash and comma as separator numbers, other symbols are ignored.
        Example (no space after comma): 1,2,5-11

    -ir [indexes], --ignore-row=[indexes]
        Rows in input files that should be ignored.
        Use only dash and comma as separator numbers, other symbols are ignored.
        1st header row always set as ignored.
        Example (no space after comma): 2,4-7

    -sep [TAB|SPACE|separator_char], --separator=[TAB|SPACE|separator_char]
        Column's separator in raw data files.
        It can be TAB or SPACE abbreviation, comma, dot, semicolon or other char.
        TAB symbol by default.

    --no-fuzzy
        Add key if You doesn't want show fuzzy results, only real. Not set by default.

    --reload
        Add key if You want reload network from file before usage. Not set by default.

    -u [epochs], --update=[epochs]
        Update error status after this epochs time, 5 by default.
        This parameter affected training speed.

*Work modes:*

Learning Mode:
    
    --learn [Network_Options]
        Start program in learning mode, where Network_Options is a dictionary:
        
        {
        config=inputs,layer1,layer2,...,outputs
            where inputs is number of neurons in input layer,
            layer1..N are number of neurons in hidden layers,
            and outputs is number of neurons in output layer

        epochs=[int_num]
            this is a positive integer number, greater than 0, means the number of training cycles

        rate=[float_num]
            this is parameter of rate of learning, float number in (0, 1]

        momentum=[float_num]
            this is parameter of momentum of learning, float number in (0, 1]

        epsilon=[float_num]
            this parameter used to compare the distance between the two vectors, float number in (0, 1]

        stop=[float_num]
            this is stop parameter of learning (percent of errors), float number in [0, 100]
        }

Classifying Mode:

    --classify [Network_Options]
        Start program in classificator mode, where Network_Options is a dictionary:

        {
        config=inputs,layer1,layer2,...,outputs
            where inputs is number of neurons in input layer,
            layer1..N are number of neurons in hidden layers,
            and outputs is number of neurons in output layer
        }


*Examples:*

Start learning with user's ethalon data file and neuronet options Config=(3,[3,2],2), 10 epochs, 0.1 learning rate and 0.05 momentum, epsilon is 0.01 and stop learning if errors less than 5%:

    python FuzzyClassificator.py --ethalons ethalons.dat --learn config=3,3,2,2 epochs=10 rate=0.1 momentum=0.05 epsilon=0.01 stop=5 --separator=TAB --debug-level=DEBUG

Classify all candidates from file candidates.dat and show result in report.txt:

    python FuzzyClassificator.py --candidates candidates.dat --network network.xml --report report.txt --classify config=3,3,2,2 --separator=TAB --debug-level=DEBUG

Where 'python' is full path to Pyzo Python 3.3.2 interpreter.

Preparing data
--------------

**ethalons.dat**

This is default file with ethalon data set. This file contains tab-delimited data (by default) that looks like this:

    <first header line with column names> 
    and then some strings contains real or fuzzy values:
    - M input columns: <1st value><tab>...<tab><M-th value>
    - N output columns: <1st value><tab>...<tab><N-th value>
For each input vector level of membership in the class characterized by the output vector.


*Example:*

    input1  input2  input3  1st_class_output  2nd_class_output
    0.1     0.2     Min     Min               Max
    0.2     0.3     Low     Min               Max
    0.3     0.4     Med     Min               Max
    0.4     0.5     Med     Max               Min
    0.5     0.6     High    Max               Min
    0.6     0.7     Max     Max               Min

For training on this data set use --learn key with config parameter, for example:

    --learn config=3,3,2,2 

where first config parameter mean that dimension of input vector is 3, last config parameter mean that 
dimension of output vector is 2, and the middle "3,2" parameters means that neural network must be created with two hidden layers, three neurons in 1st hidden layer and two neurons in 2nd.


**candidates.dat**

This is default file with data set for classifying. This file contains tab-delimited data (by default) that looks like this:

    <first header line with column names>
    and then some strings contains real or fuzzy values:
    -  M input columns: <1st value><tab>...<tab><M-th value>


*Example:*

    input1  input2  input3
    0.12    0.32    Min
    0.32    0.35    Low
    0.54    0.57    Med
    0.65    0.68    High
    0.76    0.79    Max

To classify each of input vectors You must to use --classify key. All columns are used as values of input vectors.

If You train Neuronet with command:

    python FuzzyClassificator.py --ethalons ethalons.dat --learn config=3,3,2,2 epochs=1000 rate=0.1 momentum=0.05

And then classificate candidates vectors with command:

    python FuzzyClassificator.py --candidates candidates.dat --network network.xml --report report.txt --classify config=3,3,2,2

Then You'll get *report.text* file with information that looks like this:

    Neuronet: C:\work\projects\FuzzyClassificator\network.xml

    FuzzyScale = {Min, Low, Med, High, Max}
        Min = <Hyperbolic(x, {'a': 8, 'c': 0, 'b': 20}), [0.0, 0.23]>
        Low = <Bell(x, {'a': 0.17, 'c': 0.34, 'b': 0.23}), [0.17, 0.4]>
        Med = <Bell(x, {'a': 0.34, 'c': 0.6, 'b': 0.4}), [0.34, 0.66]>
        High = <Bell(x, {'a': 0.6, 'c': 0.77, 'b': 0.66}), [0.6, 0.83]>
        Max = <Parabolic(x, {'a': 0.77, 'b': 0.95}), [0.77, 1.0]>

    Classification results for candidates vectors:

        Header: [input1 input2 input3]	[1st_class_output 2nd_class_output]
        ----------------------------------------------------------------------
        Input: ['0.12', '0.32', 'Min']	Output: ['Min', 'Max']
        Input: ['0.32', '0.35', 'Low']	Output: ['Low', 'High']
        Input: ['0.54', '0.57', 'Med']	Output: ['Max', 'Min']
        Input: ['0.65', '0.68', 'High']	Output: ['Max', 'Min']
        Input: ['0.76', '0.79', 'Max']	Output: ['Max', 'Min']


Work with program modules
--------------

**FuzzyClassificator.py**

This is main module which realizes user command-line interaction. Main methods are *LearningMode()* and *ClassifyingMode()* which provide similar program modes. The module provide user interface that implemented in PyBrainLearning.py.

Learning mode contain steps realized by *LearningMode()*:

1. Creating PyBrain network instance with pre-defined config parameters.
2. Parsing raw data file with ethalons.
3. Preparing PyBrain dataset.
4. Initialize empty PyBrain network for learning or reading network configuration from file.
5. Creating PyBrain trainer.
6. Starts learning and saving network configuration to file.

The *LearningMode()* method takes a dictionary with the values of the initialization parameters for the neural network training.

Classifying mode contains steps realized by *ClassifyingMode()*:

1. Creating PyBrain network instance.
2. Parsing raw data file with candidates.
3. Loading trained network from network configuration file.
4. Activating network for all candidate input vectors.
5. Interpreting results.

The *ClassifyingMode()* method only runs calculations using the trained neural network.

What are the console keys used to control the module - see above.

**PyBrainLearning.py**

This is library for work with fuzzy neural networks. You can import and re-use module in your programm if you'd like to realize own work with networks.

All routines to work with fuzzy neural networks realized in *FuzzyNeuroNetwork()* class. It contains next main methods:

- *ParseRawDataFile()* - used for parsing file with text raw data,
- *PrepareDataSet()* - used for converting parsed raw-data into PyBrain dataset format,
- *CreateNetwork()* - used for creating PyBrain network,
- *CreateTrainer()* - used for creating PyBrain trainer,
- *SaveNetwork()* - used for saving network in PyBrain xml-format,
- *LoadNetwork()* - used for loading network from PyBrain xml-format file,
- *Train()* - realize network training mechanism,
- *CreateReport()* - creates text report after classification vector-candidates.

You can import this class and use its methods in other projects.

**FuzzyRoutines.py**

Library contains some routines for work with fuzzy logic operators, fuzzy datasets and fuzzy scales.

There are some examples of working with fuzzy library after importing it. Just copying at the end of FuzzyRoutines and run it.

*Work with membership functions.*

Usage of some membership functions (uncomment one of them):

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

    print('Printing Membership function parameters: ', funct)

Calculating some function's values in [0, 1]:

    xPar = 0
    for i in range(0, 11, 1):
        xPar = (xPar + i) / 10
        res = funct.mju(xPar)  # calculate one value of MF with given parameters
        print('{}({:1.1f}, {}) = {:1.4f}'.format(funct.name, xPar, funct.parameters, res))

*Work with fuzzy set.*

    fuzzySet = FuzzySet(funct, (0., 1.))  # creating fuzzy set A = <mju_funct, support_set>
    print('Printing fuzzy set after init before changes:', fuzzySet)
    print('Defuz({}) = {:1.2f}'.format(fuzzySet.name, fuzzySet.Defuz()))

    changedMjuPars = copy.deepcopy(mjuPars)  # change parameters of membership function with deepcopy example:
    changedMjuPars['a'] = 0
    changedMjuPars['b'] = 1
    changedSupportSet = (0.5, 1)  # change support set
    fuzzySet.name = 'Changed fuzzy set'

    fuzzySet.mFunction.parameters = changedMjuPars
    fuzzySet.supportSet = changedSupportSet

    print('New membership function parameters: ', fuzzySet.mFunction.parameters)
    print('New support set: ', fuzzySet.supportSet)
    print('New value of Defuz({}) = {:1.2f}'.format(fuzzySet.name, fuzzySet.Defuz()))
    print('Printing fuzzy set after changes:', fuzzySet)

*Work with fuzzy scales.*

Fuzzy scale is an ordered set of linguistic variables that looks like this:

S = [{'name': 'name_1', 'fSet': fuzzySet_1}, {'name': 'name_2', 'fSet': fuzzySet_2}, ...]

where name is a linguistic name of fuzzy set, fSet is a user define fuzzy set of FuzzySet type.

    scale = FuzzyScale()  # intialize new fuzzy scale with default levels
    print('Printing default fuzzy scale in human-readable:', scale)

    print('Defuz() of all default levels:')
    for item in scale.levels:
        print('Defuz({}) = {:1.2f}'.format(item['name'], item['fSet'].Defuz()))

Add new fuzzy levels:

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

Change scale levels:

    scale.name = 'New Scale'
    scale.levels = [{'name': levelMin.name, 'fSet': levelMin},
                    {'name': levelMed.name, 'fSet': levelMed},
                    {'name': levelMax.name, 'fSet': levelMax}]  # add new ordered set of linguistic variables into scale

    print('Changed List of levels as objects:', scale.levels)
    print('Printing changed fuzzy scale in human-readable:', scale)

    print('Defuz() of all New Scale levels:')
    for item in scale.levels:
        print('Defuz({}) = {:1.2f}'.format(item['name'], item['fSet'].Defuz()))

*Work with Universal Fuzzy Scale.*

Universal fuzzy scales S_f = {Min, Low, Med, High, Max} pre-defined in UniversalFuzzyScale() class.

    uniFScale = UniversalFuzzyScale()
    print('Levels of Universal Fuzzy Scale:', uniFScale.levels)
    print('Printing scale:', uniFScale)

    print('Defuz() of all Universal Fuzzy Scale levels:')
    for item in uniFScale.levels:
        print('Defuz({}) = {:1.2f}'.format(item['name'], item['fSet'].Defuz()))

Use Fuzzy() function to looking for level on Fuzzy Scale:

    xPar = 0
    for i in range(0, 10, 1):
        xPar = (xPar + i) / 10
        res = uniFScale.Fuzzy(xPar)  # calculate fuzzy level for some real values
        print('Fuzzy({:1.1f}, {}) = {}, {}'.format(xPar, uniFScale.name, res['name'], res['fSet']))

Finding fuzzy level using GetLevelByName() function with exact matching:

    print('Finding level by name with exact matching:')

    res = uniFScale.GetLevelByName('Min')
    print('GetLevelByName(Min, {}) = {}, {}'.format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('High')
    print('GetLevelByName(High, {}) = {}, {}'.format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

    res = uniFScale.GetLevelByName('max')
    print('GetLevelByName(max, {}) = {}, {}'.format(uniFScale.name, res['name'] if res else 'None', res['fSet'] if res else 'None'))

Finding fuzzy level using GetLevelByName() function without exact matching:

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

*Work with fuzzy logic operators.*

Checks that number is in [0, 1]:

    print('IsCorrectFuzzyNumberValue(0.5) =', IsCorrectFuzzyNumberValue(0.5))
    print('IsCorrectFuzzyNumberValue(1.1) =', IsCorrectFuzzyNumberValue(1.1))

Calculates result of fuzzy NOT, fuzzy NOT with alpha parameter and parabolic fuzzy NOT operations:

    print('FNOT(0.25) =', FuzzyNOT(0.25))
    print('FNOT(0.25, alpha=0.25) =', FuzzyNOT(0.25, alpha=0.25))
    print('FNOT(0.25, alpha=0.75) =', FuzzyNOT(0.25, alpha=0.75))
    print('FNOT(0.25, alpha=1) =', FuzzyNOT(0.25, alpha=1))

    print('FNOTParabolic(0.25, alpha=0.25) =', FuzzyNOTParabolic(0.25, alpha=0.25))
    print('FNOTParabolic(0.25, alpha=0.75) =', FuzzyNOTParabolic(0.25, alpha=0.75))

Calculates result of fuzzy AND/OR operations:

    print('FuzzyAND(0.25, 0.5) =', FuzzyAND(0.25, 0.5))
    print('FuzzyOR(0.25, 0.5) =', FuzzyOR(0.25, 0.5))

Calculates result of T-Norm operations, where T-Norm is one of conjunctive operators - logic, algebraic, boundary, drastic:

    print("TNorm(0.25, 0.5, 'logic') =", TNorm(0.25, 0.5, normType='logic'))
    print("TNorm(0.25, 0.5, 'algebraic') =", TNorm(0.25, 0.5, normType='algebraic'))
    print("TNorm(0.25, 0.5, 'boundary') =", TNorm(0.25, 0.5, normType='boundary'))
    print("TNorm(0.25, 0.5, 'drastic') =", TNorm(0.25, 0.5, normType='drastic'))

Calculates result of S-coNorm operations, where S-coNorm is one of disjunctive operators - logic, algebraic, boundary, drastic:

    print("SCoNorm(0.25, 0.5, 'logic') =", SCoNorm(0.25, 0.5, normType='logic'))
    print("SCoNorm(0.25, 0.5, 'algebraic') =", SCoNorm(0.25, 0.5, normType='algebraic'))
    print("SCoNorm(0.25, 0.5, 'boundary') =", SCoNorm(0.25, 0.5, normType='boundary'))
    print("SCoNorm(0.25, 0.5, 'drastic') =", SCoNorm(0.25, 0.5, normType='drastic'))

Calculates result of T-Norm operations for N numbers, N > 2:

    print("TNormCompose(0.25, 0.5, 0.75, 'logic') =", TNormCompose(0.25, 0.5, 0.75, normType='logic'))
    print("TNormCompose(0.25, 0.5, 0.75, 'algebraic') =", TNormCompose(0.25, 0.5, 0.75, normType='algebraic'))
    print("TNormCompose(0.25, 0.5, 0.75, 'boundary') =", TNormCompose(0.25, 0.5, 0.75, normType='boundary'))
    print("TNormCompose(0.25, 0.5, 0.75, 'drastic') =", TNormCompose(0.25, 0.5, 0.75, normType='drastic'))

Calculates result of S-coNorm operations for N numbers, N > 2:

    print("SCoNormCompose(0.25, 0.5, 0.75, 'logic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='logic'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'algebraic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='algebraic'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'boundary') =", SCoNormCompose(0.25, 0.5, 0.75, normType='boundary'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'drastic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='drastic'))

If you run code above - you'll see next console output:

    Printing Membership function with parameters:  Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8})
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.0000
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.0000
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.2750
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.5525
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.8302
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 1.0000
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 1.0000
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 1.0000
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.6173
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.0617
    Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}) = 0.0000

    Printing fuzzy set after init and before changes: FuzzySet = <Trapezium(x, {"a": 0.1, "b": 1, "c": 0.5, "d": 0.8}), [0.0, 1.0]>
    Defuz(FuzzySet) = 0.59

    New membership function with parameters:  Trapezium(x, {"a": 0, "b": 1, "c": 0.5, "d": 0.8})
    New support set:  (0.5, 1)
    New value of Defuz(Changed fuzzy set) = 0.70

    Printing fuzzy set after changes: Changed fuzzy set = <Trapezium(x, {"a": 0, "b": 1, "c": 0.5, "d": 0.8}), [0.5, 1]>
    Printing default fuzzy scale in human-readable: DefaultScale = {Min, Med, High}
        Minimum = <Hyperbolic(x, {"a": 7, "b": 4, "c": 0}), [0.0, 1.0]>
        Medium = <Bell(x, {"a": 0.35, "b": 0.5, "c": 0.6}), [0.0, 1.0]>
        High = <Triangle(x, {"a": 0.7, "b": 1, "c": 1}), [0.0, 1.0]>

    Defuz() of all default levels:
    Defuz(Min) = 0.10
    Defuz(Med) = 0.55
    Defuz(High) = 0.90

    Define some new levels:
    Printing Level 1 in human-readable: min = <Hyperbolic(x, {"a": 2, "b": 20, "c": 0}), [0.0, 0.5]>
    Printing Level 2 in human-readable: med = <Bell(x, {"a": 0.4, "b": 0.55, "c": 0.7}), [0.25, 0.75]>
    Printing Level 3 in human-readable: max = <Triangle(x, {"a": 0.65, "b": 1, "c": 1}), [0.7, 1.0]>

    Changed List of levels as objects: [{'name': 'min', 'fSet': <__main__.FuzzySet object at 0x00000000027B1208>}, {'name': 'med', 'fSet': <__main__.FuzzySet object at 0x00000000027B1278>}, {'name': 'max', 'fSet': <__main__.FuzzySet object at 0x00000000027B1320>}]

    Printing changed fuzzy scale in human-readable: New Scale = {min, med, max}
        min = <Hyperbolic(x, {"a": 2, "b": 20, "c": 0}), [0.0, 0.5]>
        med = <Bell(x, {"a": 0.4, "b": 0.55, "c": 0.7}), [0.25, 0.75]>
        max = <Triangle(x, {"a": 0.65, "b": 1, "c": 1}), [0.7, 1.0]>

    Defuz() of all New Scale levels:
    Defuz(min) = 0.24
    Defuz(med) = 0.61
    Defuz(max) = 0.89

    Levels of Universal Fuzzy Scale: [{'name': 'Min', 'fSet': <__main__.FuzzySet object at 0x00000000027B14A8>}, {'name': 'Low', 'fSet': <__main__.FuzzySet object at 0x00000000027B1518>}, {'name': 'Med', 'fSet': <__main__.FuzzySet object at 0x00000000027B1588>}, {'name': 'High', 'fSet': <__main__.FuzzySet object at 0x00000000027B15F8>}, {'name': 'Max', 'fSet': <__main__.FuzzySet object at 0x00000000027B1668>}]
    Printing scale: FuzzyScale = {Min, Low, Med, High, Max}
        Min = <Hyperbolic(x, {"a": 8, "b": 20, "c": 0}), [0.0, 0.23]>
        Low = <Bell(x, {"a": 0.17, "b": 0.23, "c": 0.34}), [0.17, 0.4]>
        Med = <Bell(x, {"a": 0.34, "b": 0.4, "c": 0.6}), [0.34, 0.66]>
        High = <Bell(x, {"a": 0.6, "b": 0.66, "c": 0.77}), [0.6, 0.83]>
        Max = <Parabolic(x, {"a": 0.77, "b": 0.95}), [0.77, 1.0]>

    Defuz() of all Universal Fuzzy Scale levels:
    Defuz(Min) = 0.06
    Defuz(Low) = 0.29
    Defuz(Med) = 0.50
    Defuz(High) = 0.71
    Defuz(Max) = 0.93

    Fuzzy(0.0, FuzzyScale) = Min, Min = <Hyperbolic(x, {"a": 8, "b": 20, "c": 0}), [0.0, 0.23]>
    Fuzzy(0.1, FuzzyScale) = Min, Min = <Hyperbolic(x, {"a": 8, "b": 20, "c": 0}), [0.0, 0.23]>
    Fuzzy(0.2, FuzzyScale) = Low, Low = <Bell(x, {"a": 0.17, "b": 0.23, "c": 0.34}), [0.17, 0.4]>
    Fuzzy(0.3, FuzzyScale) = Low, Low = <Bell(x, {"a": 0.17, "b": 0.23, "c": 0.34}), [0.17, 0.4]>
    Fuzzy(0.4, FuzzyScale) = Med, Med = <Bell(x, {"a": 0.34, "b": 0.4, "c": 0.6}), [0.34, 0.66]>
    Fuzzy(0.5, FuzzyScale) = Med, Med = <Bell(x, {"a": 0.34, "b": 0.4, "c": 0.6}), [0.34, 0.66]>
    Fuzzy(0.7, FuzzyScale) = High, High = <Bell(x, {"a": 0.6, "b": 0.66, "c": 0.77}), [0.6, 0.83]>
    Fuzzy(0.8, FuzzyScale) = High, High = <Bell(x, {"a": 0.6, "b": 0.66, "c": 0.77}), [0.6, 0.83]>
    Fuzzy(0.9, FuzzyScale) = Max, Max = <Parabolic(x, {"a": 0.77, "b": 0.95}), [0.77, 1.0]>
    Fuzzy(1.0, FuzzyScale) = Max, Max = <Parabolic(x, {"a": 0.77, "b": 0.95}), [0.77, 1.0]>

    Finding level by name with exact matching:
    GetLevelByName(Min, FuzzyScale) = Min, Min = <Hyperbolic(x, {"a": 8, "b": 20, "c": 0}), [0.0, 0.23]>
    GetLevelByName(High, FuzzyScale) = High, High = <Bell(x, {"a": 0.6, "b": 0.66, "c": 0.77}), [0.6, 0.83]>
    GetLevelByName(max, FuzzyScale) = None, None

    Finding level by name without exact matching:
    GetLevelByName('mIn', FuzzyScale) = Min, Min = <Hyperbolic(x, {"a": 8, "b": 20, "c": 0}), [0.0, 0.23]>
    GetLevelByName('max', FuzzyScale) = Max, Max = <Parabolic(x, {"a": 0.77, "b": 0.95}), [0.77, 1.0]>
    GetLevelByName('Hig', FuzzyScale) = High, High = <Bell(x, {"a": 0.6, "b": 0.66, "c": 0.77}), [0.6, 0.83]>
    GetLevelByName('LOw', FuzzyScale) = Low, Low = <Bell(x, {"a": 0.17, "b": 0.23, "c": 0.34}), [0.17, 0.4]>
    GetLevelByName('eD', FuzzyScale) = Med, Med = <Bell(x, {"a": 0.34, "b": 0.4, "c": 0.6}), [0.34, 0.66]>
    GetLevelByName('Highest', FuzzyScale) = None, None

    IsCorrectFuzzyNumberValue(0.5) = True
    IsCorrectFuzzyNumberValue(1.1) = False

    FNOT(0.25) = 0.75
    FNOT(0.25, alpha=0.25) = 0.25

    FNOT(0.25, alpha=0.75) = 0.9166666666666666
    FNOT(0.25, alpha=1) = 1.0

    FNOTParabolic(0.25, alpha=0.25) = 0.25000000000000017
    FNOTParabolic(0.25, alpha=0.75) = 0.9820000000000008

    FuzzyAND(0.25, 0.5) = 0.25
    FuzzyOR(0.25, 0.5) = 0.5

    TNorm(0.25, 0.5, 'logic') = 0.25
    TNorm(0.25, 0.5, 'algebraic') = 0.125
    TNorm(0.25, 0.5, 'boundary') = 1
    TNorm(0.25, 0.5, 'drastic') = 0

    SCoNorm(0.25, 0.5, 'logic') = 0.5
    SCoNorm(0.25, 0.5, 'algebraic') = 0.625
    SCoNorm(0.25, 0.5, 'boundary') = 0.75
    SCoNorm(0.25, 0.5, 'drastic') = 1

    TNormCompose(0.25, 0.5, 0.75, 'logic') = 0.25
    TNormCompose(0.25, 0.5, 0.75, 'algebraic') = 0.09375
    TNormCompose(0.25, 0.5, 0.75, 'boundary') = 0.75
    TNormCompose(0.25, 0.5, 0.75, 'drastic') = 0

    SCoNormCompose(0.25, 0.5, 0.75, 'logic') = 0.75
    SCoNormCompose(0.25, 0.5, 0.75, 'algebraic') = 0.90625
    SCoNormCompose(0.25, 0.5, 0.75, 'boundary') = 0
    SCoNormCompose(0.25, 0.5, 0.75, 'drastic') = 1

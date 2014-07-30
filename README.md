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

FuzzyClassificator using Pyzo, http://www.pyzo.org - free and open-source computing environment, based on Python 3.3.2 and includes many scientific packages, PyBrain library, http://pybrain.org - neural network routines.


**Usage:**

    python FuzzyClassificator.py [options] [learn [Network_Options**]]|[classify]


*Optional arguments:*

    -h, --help
        Show help message and exit.

    -l <verbosity>, --debug-level <verbosity>
        Use 0, 1, 2 debug info verbosity, 0 by default.

    -e <ethalon_filename>, --ethalons <ethalon_filename>
        File with ethalon data samples, ethalons.dat by default.

    -c <candidates_filename>, --candidates <candidates_filename>
        File with candidates data samples, candidates.dat by default.

    -n <network_filename>, --network <network_filename>
        File with Neuro Network configuration, network.xml by default.

    -r <report_filename>, --report <report_filename>
        File with Neuro Network configuration, report.txt by default.

    --reload
        Reload network from file before usage, False by default.


*Work modes:*

    Some keys:
    
    --learn [Network_Options]
        Start program in learning mode with options parameters, where Network_Options** is a dictionary:
        
        {
        config=inputs,layer1,layer2,...,outputs
            where inputs is number of neurons in input layer,
            layer1..N are number of neurons in hidden layers,
            and outputs is number of neurons in output layer

        epochs=<int_num>
            this is a positive integer number, greater than 0, means the number of training cycles

        rate=<float_num>
            this is parameter of rate of learning, float number in [0, 1]

        momentum=<float_num>
            this is parameter of momentum of learning, float number in [0, 1]
        }

    --classify
        Start program in classificator mode.


*Examples:*

Start learning with user's ethalon data file and neuronet options Config=<3,[3,2],2>, 10 epochs, 0.1 learning rate and 0.05 momentum:

    python FuzzyClassificator.py --ethalons user_ethalons.dat --learn config=3,3,2,2 epochs=10 rate=0.1 momentum=0.05

Classify all candidates from file user_candidates.dat and show result in user_report.txt:

    python FuzzyClassificator.py --candidates user_candidates.dat --network user_network.xml --report user_report.txt --classify

Where 'python' is Pyzo Python 3.3.2 interpreter.

Preparing data
--------------

**ethalons.dat**

This is default file with ethalon data set. This file contains tab-delimited data that looks like this:

    <first header line with column names> 
    and then some strings contains real or fuzzy values:
    - M input columns: <1st value><tab>...<tab><M-th value>
    - N output columns: <1st value><tab>...<tab><N-th value>
For each input vector level of membership in the class characterized by the output vector.


*Example:*

    input1  input2  input3  1st_class_output  2nd_class_output
    0.1     0.2     Min     0                 Max
    0.2     0.3     Low     0                 Max
    0.3     0.4     Med     0                 Max
    0.4     0.5     Med     Max               0
    0.5     0.6     High    Max               0
    0.6     0.7     Max     Max               0

For training on this data use --learn key with config parameter, for example:

    --learn config=3,3,2,2 

where first config parameter mean that dimension of input vector is 3, last config parameter mean that 
dimension of output vector is 2, and the middle "3,2" parameters means that neural network must be created with two hidden layers, three neurons in 1st hidden layer and two neurons in 2nd.


**candidates.dat**

This is default file with data set for classifying. This file contains tab-delimited data that looks like this:

    <first header line with column names>
    and then some strings contains real or fuzzy values:
    -  M input columns: <1st value><tab>...<tab><M-th value>


*Example:*

    input1  input2  input3
    0.12    0.32    Med
    0.32    0.35    Low
    0.54    0.57    Med
    0.65    0.68    High
    0.76    0.79    Min

To classify each of input vectors using --classify key. All columns are used as values of input vectors.


Work with program modules
--------------

**FuzzyClassificator.py**

This is main module which realizes user command-line interaction. Main methods are *LearningMode()* and *ClassifyingMode()* which provide similar program modes. The module provide user interface that implemented in PyBrainLearning.py.

Learning mode contain steps in *LearningMode()*:

1. Creating PyBrain network instance with pre-defined config parameters.
2. Parsing raw data file with ethalons.
3. Preparing PyBrain dataset.
4. Initialize empty PyBrain network for learning or reading network configuration from file.
5. Creating PyBrain trainer.
6. Starts learning and saving network configuration to file.

The *LearningMode()* method takes a dictionary with the values of the initialization parameters for the neural network training.

Classifying mode contains steps in *ClassifyingMode()*:

1. Creating PyBrain network instance.
2. Parsing raw data file with candidates.
3. Preparing PyBrain dataset.
4. Loading trained network from network configuration file.
5. Activating network for all candidate input vectors.
6. Interpreting results.

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
- *CreateReport()* - creating text report after classifies vector-candidates.

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

Iniversal fuzzy scales S_f = {Min, Low, Med, High, Max} pre-defined in UniversalFuzzyScale() class.

    uniFScale = UniversalFuzzyScale()
    print('Levels of Universal Fuzzy Scale:', uniFScale.levels)
    print('Printing scale:', uniFScale)

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

check that number is in [0. 1]:

    print('IsCorrectFuzzyNumberValue(0.5) =', IsCorrectFuzzyNumberValue(0.5))
    print('IsCorrectFuzzyNumberValue(1.1) =', IsCorrectFuzzyNumberValue(1.1))

    print('FNOT(0.25) =', FuzzyNOT(0.25))
    print('FNOT(0.25, alpha=0.25) =', FuzzyNOT(0.25, alpha=0.25))
    print('FNOT(0.25, alpha=0.75) =', FuzzyNOT(0.25, alpha=0.75))
    print('FNOT(0.25, alpha=1) =', FuzzyNOT(0.25, alpha=1))

    print('FNOTParabolic(0.25, alpha=0.25) =', FuzzyNOTParabolic(0.25, alpha=0.25))
    print('FNOTParabolic(0.25, alpha=0.75) =', FuzzyNOTParabolic(0.25, alpha=0.75))

    print('FuzzyAND(0.25, 0.5) =', FuzzyAND(0.25, 0.5))
    print('FuzzyOR(0.25, 0.5) =', FuzzyOR(0.25, 0.5))

    print("TNorm(0.25, 0.5, 'logic') =", TNorm(0.25, 0.5, normType='logic'))
    print("TNorm(0.25, 0.5, 'algebraic') =", TNorm(0.25, 0.5, normType='algebraic'))
    print("TNorm(0.25, 0.5, 'boundary') =", TNorm(0.25, 0.5, normType='boundary'))
    print("TNorm(0.25, 0.5, 'drastic') =", TNorm(0.25, 0.5, normType='drastic'))

    print("SCoNorm(0.25, 0.5, 'logic') =", SCoNorm(0.25, 0.5, normType='logic'))
    print("SCoNorm(0.25, 0.5, 'algebraic') =", SCoNorm(0.25, 0.5, normType='algebraic'))
    print("SCoNorm(0.25, 0.5, 'boundary') =", SCoNorm(0.25, 0.5, normType='boundary'))
    print("SCoNorm(0.25, 0.5, 'drastic') =", SCoNorm(0.25, 0.5, normType='drastic'))

    print("TNormCompose(0.25, 0.5, 0.75, 'logic') =", TNormCompose(0.25, 0.5, 0.75, normType='logic'))
    print("TNormCompose(0.25, 0.5, 0.75, 'algebraic') =", TNormCompose(0.25, 0.5, 0.75, normType='algebraic'))
    print("TNormCompose(0.25, 0.5, 0.75, 'boundary') =", TNormCompose(0.25, 0.5, 0.75, normType='boundary'))
    print("TNormCompose(0.25, 0.5, 0.75, 'drastic') =", TNormCompose(0.25, 0.5, 0.75, normType='drastic'))

    print("SCoNormCompose(0.25, 0.5, 0.75, 'logic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='logic'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'algebraic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='algebraic'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'boundary') =", SCoNormCompose(0.25, 0.5, 0.75, normType='boundary'))
    print("SCoNormCompose(0.25, 0.5, 0.75, 'drastic') =", SCoNormCompose(0.25, 0.5, 0.75, normType='drastic'))
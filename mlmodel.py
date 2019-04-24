import random
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from fincore.format import FeatureData


class Helpers():
    @classmethod
    def thresholds(cls, model, percentile):
        ''' Compute the upper and lower percentile thresholds for the given data,
            so we can know when to buy when we are running a simulation
        '''
        predictions = model.predict(model.x_train)
        upper, lower  = np.percentile(predictions, [percentile, 100. - percentile])

        print 'Found Lower/Upper Thresholds: %.5f, %.5f' % (lower, upper)
        return lower, upper

    @classmethod
    def getmask(cls, model, percentile, inputs):
        ''' Get an index mask of the model predictions that fall above the given
            percentile
        '''
        if percentile is None:
            items = len(model.x_test)
            return np.array(range(items))
        else:
            _, upper = cls.thresholds(model, percentile)
            predictions = model.predict(inputs)
            return np.where(predictions > upper)



class MLModel(object):
    ''' Main Parent Class for Machine Learning Model Classes. The user simply
        needs to override the 'train' method.

        Properties:
            - symbol
            - x_train, y_train
            - x_test, y_test
            - features
        Methods:
            - score: R^2 test score
            - trainscore: R^2 train score
            - compare: Compares score v trainscore
            - predict: Predicts outputs given some inputs
            - save: Saves model class to local .cache folder
            - simulate: Runs a trading simulation on the test (or train) data
            - reload: Reloads feature data
            - reshape: Reshapes & reshuffles the train/test data split
            - shuffle: Shuffles the training/testing data
        Classmethods:
            - load: Attempts to load model class from local .cache folder


        eg.
        class GreatModel(MLModel):
            def train(self, **kwargs):
                self.model = GreatModelRegressor(**kwargs)
                self.model.train(self.x_train, self.y_train)

        model = GreatModel('SPY')
        model.train(alpha=0.0, beta=1.0)
        model.simulate()
        model.save()
    '''

    def __init__(self, symbol, getvix=False, train=0.75):
        ''' Save the model's symbol and Load in raw feature data
        '''
        self.symbol = symbol
        self._getvix = getvix
        self._trainsize = train
        # Load feature data:
        self.reload()

    def train(self):
        ''' Should override this method, and set self.model to be the child
            class' machine learning model
        '''
        raise NotImplementedError('Must Override the %s.train method' % self.classname)

    @property
    def classname(self):
        ''' Alias for the model class name '''
        return self.__class__.__name__

    def score(self, percentile=None):
        ''' Compute the R^2 score on the testing data. You can optionally specify
            a percentile cutoff to only compute the score one data points that
            fall above a certain train-data percentile
        '''
        # Get a percentile-based mask:
        mask = Helpers.getmask(self, percentile, self.x_test)
        return self.model.score(self.x_test[mask], self.y_test[mask])

    def trainscore(self):
        ''' Compute the R^2 score on the training data
        '''
        return self.model.score(self.x_train, self.y_train)

    def accuracy(self, percentile=None, test=True):
        ''' Determine the mean accuracy of the test or train data when the model
            predicts above a certain percentile
        '''
        # Get the input/output data for assessing:
        inputs = self.x_test if test else self.x_train
        outputs = self.y_test if test else self.y_train

        # Find the inputs that show the strongest percentile of predictions:
        mask = Helpers.getmask(self, percentile, inputs)
        predictions = self.predict(inputs[mask])

        # Now we compare all instances where our model predicted a positive return
        # with the number of observed instances that had a positive return, to
        # determine the accuracy
        predictionbool = predictions > 0
        outputbool = outputs[mask] > 0
        return sum(predictionbool == outputbool) / float(len(predictionbool))

    def compare(self):
        ''' Print a comparison of the training R^2 score and the test R^2 score
        '''
        if self.model:
            print 'Train Score: %s' % self.trainscore()
            print 'Test Score: %s' % self.score()
        else:
            print 'Model has not yet been trained...'

    def predict(self, inputs):
        ''' Predict the model outputs for the given inputs
        '''
        return self.model.predict(inputs)

    def reload(self):
        ''' Force a hard reload the Feature data for this class
        '''
        # Load Feature Data:
        data = FeatureData(self.symbol, getvix=self._getvix)
        # Split Data into
        self._inputs, self._outputs, self.features = data.format()
        self.shuffle()

    def reshape(self, train=0.6):
        ''' Reshape the train/test split size, and shuffle the data with new
            partition sizes
        '''
        self._trainsize = train
        self.shuffle()

    def shuffle(self):
        ''' Shuffle test/train data, do so by selecting a random index and
            splicing a contiguous chunk of train data, and designating it as
            the training data
            - Note that the test data must be contiguous because adjacent data
            points contain redundant information, such as the -n minute lag return
        '''
        # Get the index of the testing set:
        count = self._inputs.shape[0]
        start = random.randrange(int(count * self._trainsize))
        end = int(start + count * (1. - self._trainsize))
        print 'Repartitioning test inputs as %s:%s (out of %s total)...' % (start, end, count)

        # Partition Train Sets:
        self.x_train = np.append(self._inputs[:start], self._inputs[end:], axis=0)
        self.y_train = np.append(self._outputs[:start], self._outputs[end:], axis=0)
        # Partition Test Sets:
        self.x_test = self._inputs[start:end]
        self.y_test = self._outputs[start:end]

        print 'Partitioned: %s Training Arrays, %s Testing Arrays' % (self.x_train.shape[0], self.x_test.shape[0])

    def save(self):
        ''' Save the current class data to a local file in the .cache/ subrepository
        '''
        dump(self, '.cache/%s.%s.joblib' % (self.classname.lower(), self.symbol))

    def simulate(self, percentile=None, sell=True, test=True):
        ''' Simulate the Models' Performances throughout time with the 'siminputs'
            and 'simoutputs' datasources from the

            Args:
                - percentile: using the test data, determine the nth percentile threshold
                  for executing a trade
                - sell: boolean, whether or not to allow sell/short orders

        '''
        if test:
            inputs = self.x_test
            outputs = self.y_test
        else:
            inputs = self.x_train
            outputs = self.y_train

        returns = []
        current = 100.
        count = len(inputs)

        upper = 0.
        lower = 0.
        if percentile:
            lower, upper = Helpers.thresholds(self, percentile)

        print 'Processing Backtest Results...'
        for i in range(count):
            # Periodically print the progress:
            if i % 50 == 0:
                print '%.2f%%' % (float(i) / float(count) * 100.)

            input = inputs[i:(i+1),:]
            prediction = self.predict(input)
            observation = outputs[i]
            if prediction > upper:
                # We Buy here
                current *= (1. + observation)
            elif prediction < lower and sell:
                # We Sell here
                current *= (1. - observation)
            else:
                # Do nothing
                pass
            returns.append(current)

        plt.plot(returns)
        plt.plot([100] * len(returns))
        plt.show()

    @classmethod
    def load(cls, symbol, **kwargs):
        ''' Save the current class data to a local file in the .cache/ subrepository
        '''
        try:
            return load('.cache/%s.%s.joblib' % (cls.__name__.lower(), symbol))
        except IOError:
            return cls(symbol, **kwargs)

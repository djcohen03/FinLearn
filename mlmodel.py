import random
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from fincore.xydata import XYData


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

    def ttest(self, percentile=None, test=True):
        ''' Perform a t-test to determine if this model can predict market gains
            more often than the sample mean
        '''
        # todo...
        pass

    def accuracy(forest):
        ''' Give an assessment of the accuracy
        '''
        # Plot Train Prediction Accuracy:
        plt.figure(1)
        plt.subplot(211)
        predictions = forest.predict(forest.x_train)
        diff = forest.y_train - predictions
        plt.hist(diff.tolist()[0], 100, normed=True, range=(-0.4, 0.4), color=(1, 0, 0, 0.5))
        plt.axvline(x=0, color='black', lw=1)
        plt.axvline(x=diff.mean(), color='black', lw=2)
        plt.title('Training Error: mu=%.3f, sigma=%.3f' % (
            diff.mean(),
            diff.std(),
        ))

        # Plot Test Prediction Acurracy:
        plt.subplot(212)
        predictions = forest.predict(forest.x_test)
        diff = forest.y_test - predictions
        plt.hist(diff.tolist()[0], 100, normed=True, range=(-0.4, 0.4), color=(0, 0, 1, 0.5))
        plt.axvline(x=0, color='black', lw=1)
        plt.axvline(x=diff.mean(), color='black', lw=2)
        plt.title('Test Error: mu=%.3f, sigma=%.3f' % (
            diff.mean(),
            diff.std(),
        ))

        plt.show()

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
        self.xydata = XYData(self.symbol, lookback=30, forecast=10)
        self.shuffle()

    def reshape(self, train=0.6):
        ''' Reshape the train/test split size, and shuffle the data with new
            partition sizes
        '''
        self._trainsize = train
        self.shuffle()

    def shuffle(self):
        ''' Shuffle test/train data inputs & outputs
        '''
        self.x_train, self.y_train, self.x_test, self.y_test = self.xydata.getsplit(train=self._trainsize)
        print 'Partitioned: %s Training Arrays, %s Testing Arrays' % (self.x_train.shape[0], self.x_test.shape[0])

    def save(self):
        ''' Save the current class data to a local file in the .cache/ subrepository
        '''
        dump(self, '.cache/%s.%s.joblib' % (self.classname.lower(), self.symbol))

    def simulate(self, lower=0.0, upper=0.0, sell=True, test=True):
        ''' Runs and graphs a model's predictive accuracy over time, based on
            the realized gains/losses of buying/selling over time
        '''
        # Get the input/output data for assessing:
        inputs = self.x_test if test else self.x_train
        outputs = self.y_test if test else self.y_train

        # Get predictions & trading signal booleans:
        predictions = self.predict(inputs)
        getsignal = lambda v: -1. if v < lower else (1. if v > upper else 0.)
        signals = np.array(map(getsignal, predictions))

        # Generate a PNL Time series based on the signals and realized changes:
        results = np.multiply(signals, outputs)
        pnl = (results / 100. + 1.).cumprod() * 100.

        plt.plot(pnl)
        plt.plot([100] * len(pnl))
        plt.title('PnL From %s Sample Predictions' % len(predictions))
        plt.show()

    @classmethod
    def load(cls, symbol, **kwargs):
        ''' Save the current class data to a local file in the .cache/ subrepository
        '''
        try:
            return load('.cache/%s.%s.joblib' % (cls.__name__.lower(), symbol))
        except IOError:
            return cls(symbol, **kwargs)

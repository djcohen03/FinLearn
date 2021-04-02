import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from joblib import dump, load
from fincore.xydata import XYData


class Helpers():
    @classmethod
    def thresholds(cls, model, percentile):
        ''' Compute the upper and lower percentile thresholds for the given data,
            so we can know when to buy when we are running a simulation
        '''
        predictions = model.predict(model.datasets.train.inputs)
        upper, lower  = np.percentile(predictions, [percentile, 100. - percentile])

        print 'Found Lower/Upper Thresholds: %.5f, %.5f' % (lower, upper)
        return lower, upper

    @classmethod
    def getmask(cls, model, percentile, inputs):
        ''' Get an index mask of the model predictions that fall above the given
            percentile
        '''
        if percentile is None:
            items = len(model.datasets.test.inputs)
            return np.array(range(items))
        else:
            _, upper = cls.thresholds(model, percentile)
            predictions = model.predict(inputs)
            return np.where(predictions > upper)

    @classmethod
    def ask(cls, question):
        ''' Prompts the user to answer the given y/n question, returns the response
        '''
        prompt = '%s y/N: ' % question
        response = raw_input(prompt).strip()
        return response == 'y'

    @classmethod
    def getstats(cls, baseline, results):
        ''' Get's the mean, std, and p-value (based on a two-sided t-test) for
            the given sample results
        '''
        mean = results.mean() - baseline.mean()
        std = results.std()
        count, = results.shape
        t = np.abs(mean / (std / np.sqrt(count)))
        p = stats.t.sf(t, count - 1) * 2.
        return mean, std, p


class MLModel(object):
    ''' Main Parent Class for Machine Learning Model Classes. The user simply
        needs to override the 'train' method.

        Properties:
            - symbol
            - datasets
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
                self.model.train(self.datasets.train.inputs, self.datasets.train.outputs)

        model = GreatModel('SPY')
        model.train(alpha=0.0, beta=1.0)
        model.simulate()
        model.save()
    '''

    def __init__(self, symbol, train=0.75):
        ''' Save the model's symbol and Load in raw feature data
        '''
        self.symbol = symbol
        self._trainsize = train

        # Load Train/Testing Data Sets:
        self.datasets = XYData(self.symbol, lookback=30, forecast=15)
        self.features = self.datasets.features

    def train(self):
        ''' Should override this method, and set self.model to be the child
            class' machine learning model
        '''
        raise NotImplementedError('Must Override the %s.train method' % self.classname)

    def transform(self, inputs):
        ''' Optionally override for input transformation grooming
        '''
        return inputs

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
        mask = Helpers.getmask(self, percentile, self.datasets.test.inputs)
        return self.model.score(self.datasets.test.inputs[mask], self.datasets.test.outputs[mask])

    def trainscore(self):
        ''' Compute the R^2 score on the training data
        '''
        return self.model.score(self.datasets.train.inputs, self.datasets.train.outputs)

    def plotalpha(self, lower=0., upper=0.):
        ''' Plots gains/losses accrued from predicting direction based on
            model prediction signals
        '''
        # Get predictions & trading signal booleans:
        predictions = self.predict(self.datasets.test.inputs)
        getsignal = lambda v: -1. if v < lower else (1. if v > upper else 0.)
        signals = np.array(map(getsignal, predictions))

        # Generate a PNL Time series based on the signals and realized changes:
        results = np.multiply(signals, self.datasets.test.outputs)

        # Get some summary stats:
        mean, std, p = Helpers.getstats(self.datasets.test.outputs, results)

        # Prediction Results
        presults = results[np.nonzero(results)]

        plt.figure(figsize=(14, 8))
        plt.hist(presults.tolist(), 100, normed=True, range=(-0.4, 0.4), color=(0.1, 0.1, 0.9, 0.5))
        plt.axvline(x=0, color='gray', lw=1)
        plt.axvline(x=mean, color='green', lw=1)
        plt.title('Mean: %.3f%%, Std: %.3f%%, P-Value: %.5f' % (mean, std, p))
        plt.show()

    def plotmasks(self):
        ''' Plot R^2 As a Function of Model Selectivity
        '''
        # Determine apriori upper/lower cutoffs based on observered training data:
        lower, upper  = np.percentile(self.datasets.train.outputs, [5., 95.])
        cutoffs = np.linspace(lower, upper, num=100)

        # Do test dataset model predictions for comparing against observed dataset:
        predictions = self.predict(self.datasets.test.inputs)
        observed = self.datasets.test.outputs

        # Gather Prediction Performances Across Different Selectivity Levels:
        _values = []
        _counts = []
        for cutoff in cutoffs:
            if cutoff < 0:
                items = observed[np.where(predictions < cutoff)]
                value = -items.mean()
            else:
                items = observed[np.where(predictions > cutoff)]
                value = items.mean()

            count = items.size
            _values.append(value)
            _counts.append(count)

        # Do some plotting (don't look at the code, just enjoy the nice graph):
        plt.figure(figsize=(14, 8))
        plt.plot(cutoffs, _values, color=(0, 0.7, 0, 0.7), linewidth=3, label='Avg. Return')
        width = cutoffs[1] - cutoffs[0]
        normalizer = max(_counts) / max(_values)
        plt.bar(cutoffs, _counts / normalizer, width=width, color=(0, 0, 0.7, 0.2), edgecolor=(0, 0, 0.0, 0.0), label='Freq.')
        plt.axhline(color='black')
        plt.ylabel('Avg. Return')
        plt.xlabel('Model Selectivity')
        plt.title('Model Permance Relative To Model Selectivity: %s %s' % (self.symbol, self.classname))
        plt.legend()
        plt.show()

    def feature_importances(self, top=20):
        ''' Creates a bar plot of feature importances
        '''
        try:
            # Try to use the feature_importances_ array:
            importances = self.model.feature_importances_
            topvalues = sorted(zip(importances, self.features))[-top:]
        except AttributeError:
            # If there is no feature importan array, try to use the coef_ array:
            coef = self.model.coef_
            allvalues = sorted(zip(coef, self.features), key=lambda x: abs(x[0]))
            topvalues = allvalues[-top:]

        values, features = zip(*topvalues)
        index = list(range(top))

        # Plot important features:
        plt.figure(figsize=(14, 8))
        plt.barh(index, values, color=(0, 0.7, 0, 0.7))
        plt.yticks(index, features)
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
        # Normalize Inputs With self.transform Before Passing To Model:
        transformed = self.transform(inputs)
        return self.model.predict(transformed)

    def reshape(self, train=0.6):
        ''' Reshape the train/test split size, and shuffle the data with new
            partition sizes
        '''
        self._trainsize = train
        self.shuffle()

    def shuffle(self):
        ''' Alias For self.datasets.shuffle()
        '''
        self.datasets.shuffle(split=self._trainsize)

    def save(self):
        ''' Save the current class data to a local file in the .cache/ subrepository
        '''
        dump(self, '.cache/%s.%s.joblib' % (self.classname.lower(), self.symbol))

    def simulate(self, lower=0.0, upper=0.0, sell=True, test=True):
        ''' Runs and graphs a model's predictive accuracy over time, based on
            the realized gains/losses of buying/selling over time
        '''
        # Get the input/output data for assessing:
        inputs = self.datasets.test.inputs if test else self.datasets.train.inputs
        outputs = self.datasets.test.outputs if test else self.datasets.train.outputs

        # Get predictions & trading signal booleans:
        predictions = self.predict(inputs)
        buys = np.where(predictions > upper)
        sells = np.where(predictions < lower) if test else np.array([])

        # Generate a PNL Time series based on the signals and realized changes:
        pnl = (outputs[buys] + 1.).cumprod() * 100.

        plt.figure(figsize=(14, 8))
        plt.plot(pnl)
        plt.axhline(y=100, color='black')
        plt.title('PnL From %s Sample Predictions, Selected For Predictions > %.5f (%s Dataset)' % (
            len(predictions),
            upper,
            'Test' if test else 'Train')
        )
        plt.show()

    @classmethod
    def load(cls, symbol, **kwargs):
        ''' Save the current class data to a local file in the .cache/ subrepository
        '''
        try:
            return load('.cache/%s.%s.joblib' % (cls.__name__.lower(), symbol))
        except IOError:
            return cls(symbol, **kwargs)

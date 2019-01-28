import time
import random
from fincore.format import FeatureData
from sklearn.model_selection import train_test_split
from joblib import dump, load

class MLData(object):
    ''' A Class for Loading, Caching, Formatting, and Managing the Feature data
        used in the Machine Learning Algorithms

        # Example:
        spydata = MLData('SPY')
        x_train, x_test, y_train, y_test = spydata.get()
        model.fit(x_train, y_train)
        model.score(x_test, y_test)

    '''

    def __init__(self, symbol, getvix=False, simsize=1440):
        ''' Initialize and immediately load the feature data into memory
        '''
        self.symbol = symbol
        self.getvix = getvix
        self.simsize = simsize
        self.filename = '%s.joblib' % self.symbol
        self.load()

    def get(self):
        ''' Return the training/testing data np arrays
        '''
        return (self.x_train, self.x_test, self.y_train, self.y_test)

    def shuffle(self):
        ''' Shuffle and split training/testing data
        '''
        state = random.randint(1, 10000)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs, self.outputs, random_state=state)

    def refresh(self):
        ''' Reload the Feature data for the given symbol directly from the
            database, save the result in the data/ subrepository for later
            quick loading
        '''
        self.featuredata = FeatureData(self.symbol, getvix=self.getvix)
        self.inputs, self.outputs, self.features = self.featuredata.format()

        # Cut off a 3-day sample for simulation testing
        self.siminputs = self.inputs[-self.simsize:]
        self.simoutputs = self.outputs[-self.simsize:]
        self.inputs = self.inputs[:-self.simsize]
        self.outputs = self.outputs[:-self.simsize]
        self.save()

    def save(self):
        ''' Save the current class data to a local file in the data/ subrepository
        '''
        dump(self, 'data/%s' % self.filename)

    def load(self):
        '''
        '''
        start = time.time()
        try:
            mldata = load('data/%s' % self.filename)
            # Copy over data from old class to new:
            self.featuredata = mldata.featuredata
            self.inputs = mldata.inputs
            self.outputs = mldata.outputs
            self.siminputs = mldata.siminputs
            self.simoutputs = mldata.simoutputs
            self.x_train = getattr(mldata, 'x_train', None)
            self.x_test = getattr(mldata, 'x_test', None)
            self.y_train = getattr(mldata, 'y_train', None)
            self.y_test = getattr(mldata, 'y_test', None)
            self.save()
            print "Loaded MLData %s in %.2fs, used data to construct new class" % (mldata, time.time() - start)
        except IOError:
            # No previous data was saved, so we will need to do a hard refresh:
            self.refresh()

if __name__ == '__main__':
    pass

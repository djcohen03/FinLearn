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

    def __init__(self, symbol):
        ''' Initialize and immediately load the feature data into memory
        '''
        self.symbol = symbol
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
        self.featuredata = FeatureData(self.symbol)
        self.inputs, self.outputs, self.features = self.featuredata.format()

        # Cut off a 3-day sample for simulation testing
        simsize = 8 * 60 * 3
        self.siminputs = self.inputs[-simsize:]
        self.simoutputs = self.outputs[-simsize:]
        self.inputs = self.inputs[:simsize]
        self.outputs = self.outputs[:simsize]
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
            print "Loaded MLData %s in %.2fs, copying data into new class..." % (mldata, time.time() - start)
            # Copy over data from old class to new:
            self.featuredata = mldata.featuredata
            self.inputs = mldata.inputs
            self.outputs = mldata.outputs
            self.siminputs = mldata.siminputs
            self.simoutputs = mldata.simoutputs
            self.x_train = mldata.x_train
            self.x_test = mldata.x_test
            self.y_train = mldata.y_train
            self.y_test = mldata.y_test
            self.save()
        except IOError:
            # No previous data was saved, so we will need to do a hard refresh:
            self.refresh()

if __name__ == '__main__':
    pass

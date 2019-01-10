import time
import random
from fincore.format import FeatureData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


class RandomForest(object):
    def __init__(self, symbol):
        ''' Load in raw feature data for the given symbol
        '''
        self.featuredata = FeatureData(symbol)
        self.inputs, self.outputs, self.features = self.featuredata.format()

        # Cut off a 3-day sample for simulation testing
        simsize = 8 * 60 * 3
        self.siminputs = self.inputs[-simsize:]
        self.inputs = self.inputs[:simsize]
        # self.simoutputs = self.outputs[-simsize:]
        self.outputs = self.outputs[:simsize]

    def train(self, n_estimators=100, max_depth=None):
        '''
        '''
        # Shuffle the training/test data:
        self.shuffle()
        state = random.randint(1, 1000)
        self.forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=state)
        self.forest.fit(self.x_train, self.y_train)
        return self

    def shuffle(self):
        # Shuffle and split training/testing data
        state = random.randint(1, 1000)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs, self.outputs, random_state=state)

    def score(self):
        return self.forest.score(self.x_test, self.y_test)

    def simulate(self):
        # Todo: implement trading logic on inputs
        self.siminputs
        self.simoutputs

    def save(self):
        dump(self, 'forest.joblib')

    @classmethod
    def load(self):
        start = time.time()
        forest = load('forest.joblib')
        print "Loaded RandomForest %s in %.2fs" % (forest, time.time() - start)
        return forest

if __name__ == '__main__':
    forest = RandomForest('SPY')
    forest.train(n_estimators=10000, max_depth=15)
    print "Score: %s" % forest.score()
    forest.save()


    # Later...
    forest = RandomForest.load()
    print forest.score()

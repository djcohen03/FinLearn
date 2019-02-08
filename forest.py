import time
import random
from mldata import MLData
from simulate import Simulate
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

class RandomForest(object):
    def __init__(self, symbol, getvix=False, simsize=1440):
        ''' Load in raw feature data for the given symbol
        '''
        self.symbol = symbol
        self.data = MLData(symbol, getvix=getvix, simsize=simsize)

    def train(self, n_estimators=100, max_depth=15):
        '''
        '''
        print 'Training %s-Estimator Random Forest...' % n_estimators
        start = time.time()

        # Shuffle the training/test data:
        self.data.shuffle()
        state = random.randint(1, 1000)
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=state)
        self.model.fit(self.data.x_train, self.data.y_train)
        print 'Fitted a %s-Estimator Forest in %.2fs' % (n_estimators, time.time() - start)
        return self

    def score(self):
        return self.model.score(self.data.x_test, self.data.y_test)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def simulate(self, percentile=None):
        ''' Run a small simulation
        '''
        Simulate.run(self, percentile=percentile)

    def save(self):
        ''' Save the current class data to a local file in the data/ subrepository
        '''
        dump(self, '.cache/forest.joblib')

    @classmethod
    def load(cls):
        ''' Save the current class data to a local file in the data/ subrepository
        '''
        return load('.cache/forest.joblib')


if __name__ == '__main__':
    #
    forest = RandomForest('SPY', getvix=False, simsize=5000)
    forest.train(n_estimators=50, max_depth=50)
    print "Score: %s" % forest.score()
    # forest.save()
    forest.simulate()

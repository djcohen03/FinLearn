import time
import random
from mldata import MLData
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


class RandomForest(object):
    def __init__(self, symbol):
        ''' Load in raw feature data for the given symbol
        '''
        self.symbol = symbol
        self.data = MLData(symbol)

    def train(self, n_estimators=100, max_depth=None):
        '''
        '''
        start = time.time()

        # Shuffle the training/test data:
        self.data.shuffle()
        state = random.randint(1, 1000)
        self.forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=state)
        self.forest.fit(self.data.x_train, self.data.y_train)
        print 'Fitted a %s-Estimator Forest in %.2fs' % (n_estimators, time.time() - start)
        return self

    def score(self):
        return self.forest.score(self.data.x_test, self.data.y_test)

    def predict(self, inputs):
        return self.forest.predict(inputs)

    def save(self):
        filename = 'forest[%s].joblib' % self.symbol
        dump(self, 'models/%s' % filename)
        print 'Saved Model as: %s' % filename

    @classmethod
    def load(cls, symbol):
        start = time.time()
        try:
            filename = 'forest[%s].joblib' % symbol
            forest = load('models/%s' % filename)
            print "Loaded RandomForest %s in %.2fs" % (forest, time.time() - start)
            return forest
        except IOError:
            print 'No Random Forest Available, Creating New Instance Now...'
            return cls(symbol)

if __name__ == '__main__':
    forest = RandomForest.load('SPY')
    forest.train(n_estimators=50000, max_depth=15)
    forest.save()
    # if not getattr(forest, 'forest', None):
    #     print 'Training Random Forest'
    #     forest.train(n_estimators=5000, max_depth=15)
    #     forest.save()
    print "Score: %s" % forest.score()

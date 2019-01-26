import time
import random
from mldata import MLData
from sklearn.neighbors import KNeighborsRegressor
from joblib import dump, load

class KNeighbors(object):
    def __init__(self, symbol):
        ''' Load in raw feature data for the given symbol
        '''
        self.symbol = symbol
        self.data = MLData(symbol)

    def train(self, n_neighbors=5):
        '''
        '''
        # Shuffle the training/test data:
        self.data.shuffle()
        state = random.randint(1, 1000)
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.model.fit(self.data.x_train, self.data.y_train)
        return self

    def score(self):
        return self.model.score(self.data.x_test, self.data.y_test)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def save(self):
        filename = 'neighbors[%s].joblib' % self.symbol
        dump(self, 'models/%s' % filename)

    @classmethod
    def load(cls, symbol):
        try:
            start = time.time()
            filename = 'neighbors[%s].joblib' % symbol
            model = load('models/%s' % filename)
            print "Loaded Model %s in %.2fs" % (model, time.time() - start)
            return model
        except IOError:
            print 'No Model Available, Creating New Instance Now...'
            return cls(symbol)


if __name__ == '__main__':
    model = KNeighbors.load('SPY')
    model.train(n_neighbors=3)
    model.save()
    print "Score: %s" % model.score()

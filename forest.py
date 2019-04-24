import time
import random
from mlmodel import MLModel
from sklearn.ensemble import RandomForestRegressor

class RandomForest(MLModel):

    def train(self, n_estimators=100, max_depth=15, **kwargs):
        ''' Train a Random Forest Regression Model with 'n_estimators' trees,
            each with a max depth given by the 'max_depth' parameter
        '''
        print 'Training %s-Estimator Random Forest...' % n_estimators
        start = time.time()

        # Shuffle the training/test data:
        self.shuffle()
        state = random.randint(1, 1000)
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=state, **kwargs)
        self.model.fit(self.x_train, self.y_train)

        # Print a summary of the trained model:
        print 'Fitted a %s-Estimator Forest in %.2fs' % (n_estimators, time.time() - start)
        self.compare()


if __name__ == '__main__':
    #
    forest = RandomForest('SPY')
    forest.train(n_estimators=500, max_depth=15)
    forest.save()

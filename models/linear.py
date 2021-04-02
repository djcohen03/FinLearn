from mlmodel import MLModel
from sklearn.linear_model import LinearRegression, Ridge

class LinearModel(MLModel):

    def train(self, n_jobs=None):
        '''
        '''
        # Shuffle the training/test data:
        self.shuffle()

        # Create & Train Model:
        self.model = LinearRegression(n_jobs=n_jobs)
        self.model.fit(self.transform(self.datasets.train.inputs), self.datasets.train.outputs)

class RidgeModel(MLModel):

    def train(self, alpha=1.0):
        '''
        '''
        # Shuffle the training/test data:
        self.shuffle()

        # Create & Train Model:
        self.model = Ridge(alpha=alpha)
        self.model.fit(self.transform(self.datasets.train.inputs), self.datasets.train.outputs)



if __name__ == '__main__':
    model = RidgeModel('SPY', train=0.5)
    model.train(alpha=1.0)

    model = LinearModel('SPY', train=0.5)
    model.train(n_jobs=-1)

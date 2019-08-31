from mlmodel import MLModel
from sklearn.neighbors import KNeighborsRegressor

class KNeighbors(MLModel):

    def train(self, n_neighbors=5):
        '''
        '''
        # Shuffle the training/test data:
        self.shuffle()
        # Create & Train Model:
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.model.fit(self.x_train, self.y_train)


if __name__ == '__main__':
    model = KNeighbors('SPY', train=0.75)
    model.train(n_neighbors=3)

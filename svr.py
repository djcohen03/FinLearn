from mlmodel import MLModel
from sklearn.svm import SVR

class SVRModel(MLModel):

    def train(self, C=1.0):
        '''
        '''
        # Shuffle the training/test data:
        self.shuffle()
        # Create & Train Model:
        self.model = SVR(C=C)
        self.model.fit(self.x_train, self.y_train)


if __name__ == '__main__':
    model = SVRModel('SPY', train=0.5)
    model.train(C=1.0)

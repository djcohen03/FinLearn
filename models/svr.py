from mlmodel import MLModel
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SVRModel(MLModel):

    def train(self, C=1.0):
        '''
        '''
        # Shuffle the training/test data:
        self.shuffle()

        # Fit an instance of the Standard Scaler to our new training data, so that
        # the features are normalized around zero.  This is hopefully a way to
        # improve the SV model accuracy:
        scaler = StandardScaler().fit(self.x_train)
        self.transform = lambda x: scaler.transform(x)

        # Create & Train Model:
        self.model = SVR(C=C)
        self.model.fit(self.transform(self.x_train), self.y_train)




if __name__ == '__main__':
    model = SVRModel('SPY', train=0.5)
    model.train(C=1.0)

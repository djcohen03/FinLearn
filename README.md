# FinLearn
`sklearn`-based repository for training machine learning models on a financial data set.

## Initialization:
See: [fincore](https://github.com/djcohen03/fincore) for initializing and collecting alphavantage API data.

## Building a Model
Building a model should be as simple as creating a model class that inherits the `MLModel` mixin, and overwritting the `self.train` method creating a `sklearn` model class and training it with the `x_train` and `y_train` data. This should be as simple as:

```
# kneighbors.py
from mlmodel import MLModel
from sklearn.neighbors import KNeighborsRegressor
class KNeighbors(MLModel):
    def train(self, n_neighbors=3):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.model.train(self.x_train, self.y_train)

if __name__ == '__main__':
    model = KNeighbors('SPY')
    model.train()
...
```


## MLModel class:
### Properties:
- Testing/Training: `x_train`, `y_train`, `x_test`, `y_test`
- Features: `features`
- Data: `xydata`
### Methods:
- `save():` Saves model class to local .cache folder
- `score():` R^2 test score
- `reload():` Reloads feature data
- `shuffle():` Shuffles the training/testing data
- `predict(inputs):` Predicts outputs given some inputs
- `reshape(train):` Reshapes & reshuffles the train/test data split
### Model Analysis:
- `simulate(lower, upper, test=True):` Simulates buying/selling on test data based on model signal
- `plotalpha(lower, upper):` Plots a histogram of buying/selling results based on model signals
- `feature_importances(top=20):` Plots the 20 most important model features, if the trained model uses the `feature_importances_` convention from `sklearn`

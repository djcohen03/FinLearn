# FinLearn
This is a `sklearn`-based repository for training machine learning models on financial data.  There are two main components:
- `fincore/`: This sub-repository initializes, manages, and collects financial market data, based on the alphavantage API
- `mlmodel.py`: This file contains the `MLModel` class, which can be subclassed to easily train and analyze an `sklearn` regressor model

## Initialization:
See the `fincore` [documentation](https://github.com/djcohen03/fincore) for initializing and collecting data through the alphavantage API.  
You will need to have some database available.

## Building a Model
To build a model, you must first import and inherit the MLModel class, and override the classes `train(...)` method, like so:
```
from mlmodel import MLModel
from sklearn.neighbors import KNeighborsRegressor
class KNeighbors(MLModel):
    def train(self, n_neighbors=3, **kwargs):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
        self.model.train(self.x_train, self.y_train)
```

Once you've initialized your model class you should be ready to build and train a predictive model:
```
model = KNeighbors('SPY')
model.train()
model.simulate()
```


## MLModel class:
These properties and methods will be available to the child model class, after it's been initialized it is was initialized above:
### Properties:
- `model.symbol`
- `model.x_train`
- `model.y_train`
- `model.x_test`
- `model.y_test`
- `model.features`
### Methods:
- `model.save()`: Saves model class to local .cache folder
- `model.score()`: R^2 test score
- `model.reload()`: Reloads feature data
- `model.shuffle()`: Shuffles the training/testing data
- `model.compare()`: Compares score v trainscore
- `model.trainscore()`: R^2 train score
- `model.predict(inputs)`: Predicts outputs given some inputs
- `model.reshape(train)`: Reshapes & reshuffles the train/test data split
- `model.simulate(percentile=None, sell=True, test=True)`: Runs a trading simulation on the test (or train) data
### Classmethods:
- `MLModel.load(symbol, **kwargs)`: Attempts to load model class from local .cache folder

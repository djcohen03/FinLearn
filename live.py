import sys
import traceback
import pandas as pd
from models import *
from mlmodel import MLModel
from fincore.xydata import LiveXYData

class LiveTrader(object):
    def __init__(self, name):
        '''
        '''
        print 'Loading Model %s...' % name
        self.model = MLModel.load(name)
        self.symbol = self.model.symbol
        self.lookback = int(self.model.features[-1].split('.')[-1]) + 1
        self.client = LiveXYData(self.symbol, self.lookback)

        # Keep a record of the predictions
        self._predictions = {}

    def run(self):
        '''
        '''
        print 'Starting Live Pricing Predictor...'
        for livepoint in self.client.livestream():
            try:
                # Make and save model prediction:
                prediction = self.model.predict(livepoint.inputs) * 100.
                self._predictions[livepoint.timestamp] = {self.symbol: prediction}
                print self.predictions
            except:
                print 'An Exception Occurred While Running Model Predictions:'
                print traceback.format_exc()

    @property
    def predictions(self):
        '''
        '''
        return pd.DataFrame(self._predictions).T


if __name__ == '__main__':
    name = '42120d83-0315-4c63-be23-79e87004ad04'
    service = LiveTrader(name)
    service.run()

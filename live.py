import time
import datetime
import requests
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style


style.use('fivethirtyeight')

class Helpers(object):
    @classmethod
    def sleepminute(cls, buffer=0.0):
        ''' Sleeps program until the start of the next UTC minute
        '''
        now = datetime.datetime.utcnow()
        sleeptime = 60. - (now.second + now.microsecond / 1000000.0) + buffer
        print 'Sleeping %.2fs...' % sleeptime
        time.sleep(sleeptime)

class LiveTrader(object):
    def __init__(self, model, key):
        ''' Class for running live predictions using the AlphaVantage API, and
            the given predictive model
        '''
        self.model = model
        self._key = key
        self.symbol = model.symbol
        self.lookback = model.xydata.lookback
        self.forecast = model.xydata.forecast
        self.datapoints = []

    def load(self):
        ''' Loads and saves the most recent price, time, and input vector
        '''
        # Get live datapoint:
        datapoint = self.model.xydata.live()
        self.datapoints.append(datapoint)
        return datapoint

    def plot(self):
        ''' Updates the self.ax1 plot to display the pnl progression
        '''
        if self.datapoints:
            # Get the % Changes and times:
            prices = [item.price for item in self.datapoints]
            changes = [np.log(price / prices[0]) for price in prices]
            times = [item.time for item in self.datapoints]

            # Compute PNL Time series based on the buy orders:
            pnl = [0.] * len(times)
            for i, datapoint in enumerate(self.datapoints):
                if datapoint.bought:
                    purchased = datapoint.price
                    pnlchange = [
                        np.log(price / purchased)
                        for price in
                        prices[i:(i + self.lookback)]
                    ]
                    for k in range():
                        pass

            # Overwrite pnl.jpg file with new data:
            plt.plot(times, changes)
            plt.savefig('pnl.jpg')

    def start(self):
        ''' Start daemonized trading process
        '''
        while True:
            # Sleep until the next minute:
            Helpers.sleepminute(buffer=0.5)

            # Load the most recent data point:
            current = self.load()

            # Predict price change based on most recent datapoint:
            inputs = self.model.transform(self.model.current.inputs)
            prediction = self.model.predict(inputs)[0]
            timediff = (datetime.datetime.now() - current.time).total_seconds()
            print 'Predicted %.4f (Prediction Warning: Inputs are %.1fs old)' % (prediction, timediff)

            # If the prediction is positive, then we buy here:
            if prediction > 0.:
                print 'Buying at %s..' % current.time
                current.bought = True

            # Save the prediction:
            current.prediction = prediction
            # Update the live plot:
            self.plot()


if __name__ == '__main__':
    # Create & Train a Linear Regression Model:
    from models import LinearModel
    model = LinearModel('SPY', train=.99)
    model.train(n_jobs=-1)

    # Create a Live trading class instance:
    trader = LiveTrader(model, API_KEY)
    trader.start()

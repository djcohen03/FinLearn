import numpy as np
import matplotlib.pyplot as plt

class Helpers():
    @classmethod
    def thresholds(cls, model, percentile):
        ''' Compute the upper and lower percentile thresholds for the given data
        '''
        print 'Computing %.2f%% Thresholds for %s...' % (percentile, model)
        predictions = model.predict(model.data.x_train)

        upper, lower  = np.percentile(predictions, [percentile, 100. - percentile])
        print 'Found Lower/Upper Thresholds: %s, %s' % (lower, upper)
        return lower, upper

class Simulate(object):
    @classmethod
    def run(cls, model, percentile=None, sell=True):
        ''' Simulate the Models' Performances throughout time with the 'siminputs'
            and 'simoutputs' datasources from the

            Args:
                - percentile: using the test data, determine the nth percentile threshold
                  for executing a trade
                - sell: boolean, whether or not to allow sell/short orders

        '''
        inputs = model.data.siminputs
        outputs = model.data.simoutputs
        returns = []
        current = 100.
        count = len(inputs)

        upper = 0.
        lower = 0.
        if percentile:
            lower, upper = Helpers.thresholds(model, percentile)

        print 'Processing Backtest Results...'
        for i in range(count):
            # Periodically print the progress:
            if i % 50 == 0:
                print '%.2f%%' % (float(i) / float(count) * 100.)

            input = inputs[i:(i+1),:]
            prediction = model.predict(input)
            observation = outputs[i]
            if prediction > upper:
                # We Buy here
                current *= (1. + observation)
            elif prediction < lower and sell:
                # We Sell here
                current *= (1. - observation)
            else:
                # Do nothing
                pass
            returns.append(current)

        plt.plot(returns)
        plt.plot([100] * len(returns))
        plt.show()

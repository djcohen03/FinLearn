import matplotlib.pyplot as plt

class Simulate(object):
    @classmethod
    def run(cls, model, periods=10, threshold=0.0):
        ''' Simulate the Models' Performances throughout time with the 'siminputs'
            and 'simoutputs' datasources from the
        '''
        inputs = model.data.siminputs
        outputs = model.data.simoutputs
        returns = []
        current = 100.
        count = len(inputs)
        print 'Processing Backtest Results...'
        for i in range(count):
            # Periodically print the progress:
            if i % 50 == 0:
                print '%.2f%%' % (float(i) / float(count) * 100.)

            input = inputs[i:(i+1),:]
            prediction = model.predict(input)
            observation = outputs[i]
            if prediction > threshold:
                # We Buy here
                current *= (1. + observation)
            elif prediction < -threshold:
                # We Sell here
                current *= (1. - observation)
            else:
                # Do nothing
                pass
            returns.append(current)

        plt.plot(returns)
        plt.plot([100] * len(returns))
        plt.show()

if __name__ == '__main__':
    from kneighbors import KNeighbors
    model = KNeighbors.load('SPY')
    Simulate.run(model)

    from forest import RandomForest
    forest = RandomForest.load('SPY')
    Simulate.run(forest)

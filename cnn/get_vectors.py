__author__ = 'haohanwang'

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LinearRegression, LogisticRegression, load_data_final
from optimizers import Optimizer

import pickle
import numpy as np
import numpy
import time

def tanh(x):
    return T.tanh(x)

def rectifier(x):
    return T.maximum(0., x)

def linear(x):
    return x

def sigmoid(x):
    return T.nnet.sigmoid(x)

class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=tanh):
        self.activation = activation
        if W is None:
            W_values = numpy.asarray(
                    rng.uniform(
                            low=-numpy.sqrt(6. / (n_in + n_out)),
                            high=numpy.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
            )
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.rng = rng

        self.params = [self.W, self.b]

    def output(self, input):
        lin_output = T.dot(input, self.W) + self.b
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, n_in, n_out,
                 activation, dropout_rate, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)
        self.dropout_rate = dropout_rate
        # if dropout_rate > 0:
        #     self.output = self.dropout_from_layer(rng, self.output, p=dropout_rate)

    def dropout_from_layer(self, layer):
        """p is the probablity of dropping a unit
        """
        srng = theano.tensor.shared_randomstreams.RandomStreams(
                self.rng.randint(999999))
        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = srng.binomial(n=1, p=1 - self.dropout_rate, size=layer.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        return output

    def output(self, input):
        lin_output = T.dot(input, self.W) + self.b

        r = self.activation(lin_output)

        if self.dropout_rate > 0:
            return self.dropout_from_layer(r)
        else:
            return r


class MLP(object):
    def __init__(self, rng, input1, input2, configs=(), n_out=2, batch_size=3, activation=linear):
        self.hiddenLayer1 = DropoutHiddenLayer(
                rng=rng,
                n_in=configs[0],
                n_out=configs[1],
                activation=activation,
                dropout_rate=-1
        )

        self.hiddenLayer2 = HiddenLayer(
                rng=rng,
                n_in=configs[1],
                n_out=configs[2],
                activation=activation
        )

        out1_a = self.hiddenLayer1.output(input1)
        out1_b = self.hiddenLayer1.output(input2)

        out2_a = self.hiddenLayer2.output(out1_a)
        out2_b = self.hiddenLayer2.output(out1_b)

        h_output = T.concatenate([out2_a, out2_b], axis=1) # todo: here is untested

        self.out1 = out2_a
        self.out2 = out2_b
        #
        # h_output = out1_a * out1_b
        #
        # covpol_input = h_output.reshape((batch_size, 1, 1, configs[1]))
        #
        # # TODO: free variables, batchsize = 20, nkerns = 4
        # self.CovPol = LeNetConvPoolLayer(
        #     rng,
        #     input=covpol_input,
        #     image_shape=(batch_size, 1, 1, configs[1]),
        #     filter_shape=(4, 1, 1, 3),
        #     poolsize=(1, 3)
        # )
        #
        # covpol_output = self.CovPol.output.flatten(2)

        self.logRegressionLayer = LogisticRegression(
                input=h_output,
                n_in=configs[2] * 2,
                n_out=n_out
        )

        # self.CovPol_GO = LeNetConvPoolLayer(
        #     rng,
        #     input=covpol_input,
        #     image_shape=(batch_size, 1, 1, configs[1]),
        #     filter_shape=(4, 1, 1, 3),
        #     poolsize=(1, 6)
        # )
        #
        # covpol_output_go = self.CovPol_GO.output.flatten(2)
        # covpol_output_go = covpol_output

        self.regressionLayer_BP = LinearRegression(
                input=h_output,
                n_in=configs[2] * 2,
                n_out=5
        )

        self.regressionLayer_MF = LinearRegression(
                input=h_output,
                n_in=configs[2] * 2,
                n_out=5
        )

        self.regressionLayer_CC = LinearRegression(
                input=h_output,
                n_in=configs[2] * 2,
                n_out=5
        )

        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params \
                      + self.logRegressionLayer.params \
                      + self.regressionLayer_BP.params + self.regressionLayer_CC.params + self.regressionLayer_MF.params \
            # + self.CovPol_GO.params
        # self.params = self.hiddenLayer1.params + self.logRegressionLayer.params

    def negative_log_likelihood(self, y):
        return self.logRegressionLayer.negative_log_likelihood(y)

    def distance(self, y):
        m = T.sum(T.square(self.out1 - self.out2), axis=1)
        s = T.dot(m, y) + 1e-3 / (T.dot(m, -(y - 1)) + 1e-3)
        # a = self.out1 - self.out1.mean()
        # b = self.out2 - self.out2.mean()
        # m = a*b/T.sqrt((a**2).sum()*(b**2).sum())
        # s = T.dot(m, y) + 1e-3/(T.dot(m, -(y-1))+1e-3)
        return s

    def mse_bp(self, y):
        return self.regressionLayer_BP.mse(y)

    def mse_mf(self, y):
        return self.regressionLayer_MF.mse(y)

    def mse_cc(self, y):
        return self.regressionLayer_CC.mse(y)

    def L1_reg_all(self, l=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)):
        assert len(l) == len(self.params)
        reg = 0
        for (r, p) in zip(l, self.params):
            reg += r * abs(p).sum()
        return reg

    def L2_reg_all(self, l=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)):
        assert len(l) == len(self.params)
        reg = 0
        for (r, p) in zip(l, self.params):
            reg += r * ((p ** 2).sum())
        return reg

    def L1_reg_lower(self, l=(0, 0, 0, 0)):
        assert len(l) == len(self.hiddenLayer1.params + self.hiddenLayer2.params)
        reg = 0
        for (r, p) in zip(l, self.hiddenLayer1.params + self.hiddenLayer2.params):
            reg += r * abs(p).sum()
        return reg

    def L2_reg_lower(self, l=(0, 0, 0, 0)):
        assert len(l) == len(self.hiddenLayer1.params + self.hiddenLayer2.params)
        reg = 0
        for (r, p) in zip(l, self.hiddenLayer1.params + self.hiddenLayer2.params):
            reg += r * ((p ** 2).sum())
        return reg

    def augment(self, params, mu, rho):
        r = 0
        for (p1, p2) in zip(self.params, params):
            r += (mu + rho) * ((p1 - p2) ** 2).sum()
        return r

def classify_mlp(batch_size=500, output_size=20):
    index = T.lscalar()
    x = T.matrix('x1')

    rng = numpy.random.RandomState(1234)

    activation = rectifier

    classifier = MLP(
            rng=rng,
            input1=x,
            input2=x,
            configs=(158, 300, 500),
            n_out=2,
            batch_size=batch_size,
            activation=activation
    )

    params = pickle.load(open('../model/mlp.1.pkl'))

    classifier.params[0].W = theano.shared(
        value=numpy.array(
            params[0].get_value(True),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

    classifier.params[0].b = theano.shared(
        value=numpy.array(
            params[1].get_value(True),
            dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
    )

    classifier.params[1].W = theano.shared(
        value=numpy.array(
            params[2].get_value(True),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

    classifier.params[1].b = theano.shared(
        value=numpy.array(
            params[3].get_value(True),
            dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
    )

    classifier.params[2].W = theano.shared(
        value=numpy.array(
            params[4].get_value(True),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

    classifier.params[2].b = theano.shared(
        value=numpy.array(
            params[5].get_value(True),
            dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
    )

    classify_set_x = load_data_final()
    n_classify_batches = classify_set_x.get_value(borrow=True).shape[0]
    n_classify_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    classify = theano.function(
            [index],
            classifier.out1,
            givens={
                x: classify_set_x[index * batch_size: (index + 1) * batch_size],
            }
        )

    r = []

    for i in xrange(n_classify_batches):
        m = classify(i)
        r.extend(m)
    r = np.array(r)
    print r.shape
    # r = np.append(r, np.reshape(classify_set_y.eval(),(dimension[k], 1)), 1)
    numpy.savetxt('../extractedInformation/information.csv', r, delimiter=",")

if __name__ == '__main__':
    classify_mlp()
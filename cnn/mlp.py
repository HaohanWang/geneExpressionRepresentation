import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data
from cnn import LeNetConvPoolLayer
from optimizers import Optimizer



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
        mask = srng.binomial(n=1, p=1-self.dropout_rate, size=layer.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        return output

    def output(self, input):
        lin_output = T.dot(input, self.W) + self.b

        r = self.activation(lin_output)

        if self.dropout_rate>0:
            return self.dropout_from_layer(r)
        else:
            return r




class MLP(object):
    def __init__(self, rng, input1, input2, configs=(), n_out=2, batch_size=3):
        self.hiddenLayer1 = DropoutHiddenLayer(
            rng=rng,
            n_in=configs[0],
            n_out=configs[1],
            activation=tanh,
            dropout_rate=-1
        )

        # self.hiddenLayer2 = HiddenLayer(
        #     rng=rng,
        #     n_in=configs[1],
        #     n_out=configs[2],
        #     activation=sigmoid
        # )

        out1_a = self.hiddenLayer1.output(input1)
        out1_b = self.hiddenLayer1.output(input2)

        # out2_a = self.hiddenLayer2.output(out1_a)
        # out2_b = self.hiddenLayer2.output(out1_b)

        # h_output = [out1_a, out1_b]
        # shp = out1_a.shape
        # h_output = T.reshape(h_output, [shp[0], shp[1] * 2])

        self.out1 = out1_a
        self.out2 = out1_b

        h_output = out1_a * out1_b

        covpol_input = h_output.reshape((batch_size, 1, 8, 8))

        # TODO: free variables, batchsize = 20, nkerns = 4
        self.CovPol = LeNetConvPoolLayer(
            rng,
            input=covpol_input,
            image_shape=(batch_size, 1, 8, 8),
            filter_shape=(4, 1, 3, 3),
            poolsize=(2, 2)
        )

        covpol_output = self.CovPol.output.flatten(2)

        self.logRegressionLayer = LogisticRegression(
            input=covpol_output,
            n_in=36,
            n_out=n_out
        )
        self.L1 = (
            abs(self.hiddenLayer1.W).sum()  # + abs(self.hiddenLayer2.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum()  # + (self.hiddenLayer2.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer1.params + self.logRegressionLayer.params
        # self.params = self.hiddenLayer1.params + self.logRegressionLayer.params

    def negative_log_likelihood(self, y):
        return self.logRegressionLayer.negative_log_likelihood(y)

    def distance(self, y):
        m = T.sum(T.square(self.out1 - self.out2), axis=1)
        s = T.dot(m, y) + 1e-3/(T.dot(m, -(y-1))+1e-3)
        # s = T.dot(m, y) - T.dot(m, -(y-1))
        return s



def test_mlp(learning_rate=0.1, L1_reg=0., L2_reg=0., D_reg=1.0, n_epochs=1000, batch_size=1000, cv=1):
    datasets = load_data(cv)

    train_set_x1, train_set_x2, train_set_y = datasets[0]
    valid_set_x1, valid_set_x2, valid_set_y = datasets[1]

    n_train_batches = train_set_x1.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x1.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    index = T.lscalar()
    x1 = T.matrix('x1')
    x2 = T.matrix('x2')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input1=x1,
        input2=x2,
        configs=(158, 64, 5),
        n_out=2,
        batch_size=batch_size
    )

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
        + D_reg* classifier.distance(y)
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=(classifier.errors(y), classifier.distance(y)),
        givens={
            x1: valid_set_x1[index * batch_size:(index + 1) * batch_size],
            x2: valid_set_x2[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    opt = Optimizer()
    updates = opt.adagrad(cost, classifier.params, learning_rate)

    # gparams = [T.grad(cost, param) for param in classifier.params]
    # updates = [
    #     (param, param - learning_rate * gparam)
    #     for param, gparam in zip(classifier.params, gparams)
    #     ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x1: train_set_x1[index * batch_size: (index + 1) * batch_size],
            x2: train_set_x2[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print '... training'

    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_validation_distance = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    # best_script = open('mlp.py')
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                validation_loss, validation_distance = zip(*validation_losses)
                this_validation_loss = numpy.mean(validation_loss)
                this_validation_distance = numpy.mean(validation_distance)


                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%, validation distance %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.,
                        this_validation_distance
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if (this_validation_loss < best_validation_loss *improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_score = best_validation_loss

                    print '----BEST Validated MODEL here', test_score * 100, '%----'

                if this_validation_distance < best_validation_distance:
                    if (this_validation_distance < best_validation_distance *improvement_threshold):
                        patience = max(patience, iter * patience_increase)
                    # f = open('../model/best_script_' + str(cv) + '.py', 'w')
                    # f.writelines(best_script)
                    # f.close()

            if patience <= iter:
                done_looping = True
                break
        print minibatch_avg_cost
        print classifier.params[-1].get_value(True)
        # print classifier.params[1].get_value(True)

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    # import sys
    # args = sys.argv
    # lr = float(args[1])
    # batch_size = int(args[2])
    lr = 0.001
    batch_size = 400
    l1 = 0
    l2 = 1e-4
    dr = 1e-6
    test_mlp(cv=1, learning_rate=lr, L1_reg=l1, L2_reg=l2, D_reg=dr, batch_size=batch_size)

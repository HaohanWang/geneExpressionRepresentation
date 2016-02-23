"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

filePath = '/media/haohanwang/DATA/BEST of Best/State Of Art/PPI4/NetworkHuman2/data/'

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class LinearRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.dot(input, self.W) + self.b
        self.params = [self.W, self.b]
        self.input = input

    def mse(self, y, w=None):
        if w is None:
            return ((self.y_pred - y)**2).mean()
        else:
            return ((T.dot(w, (self.y_pred - y)))**2).mean()  #TODO check here, not sure if it makes sense

def load_data(cv=1, weight=False):
    def shared_dataset(data_x_l, data_y, borrow=True):
        shared_x_L = []
        for data_x in data_x_l:
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_x_L.append(shared_x)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x_L, T.cast(shared_y, 'int32')

    def splitDataSet(data, D2=True):
        if D2:
            return data[::2,:],data[1::2,:]
        else:
            return data[::2], data[1::2]


    trda = numpy.loadtxt(filePath + 'split/data_train_'+str(cv)+'_a.txt', delimiter=',')
    trdb = numpy.loadtxt(filePath + 'split/data_train_'+str(cv)+'_b.txt', delimiter=',')

    da = numpy.loadtxt(filePath + 'split/data_test_'+str(cv)+'_a.txt', delimiter=',')
    db = numpy.loadtxt(filePath + 'split/data_test_'+str(cv)+'_b.txt', delimiter=',')
    teda, vda = splitDataSet(da, D2=True)
    tedb, vdb = splitDataSet(db, D2=True)

    trl = [int(line.strip()) for line in open(filePath + 'split/labels_train_'+str(cv)+'.txt')]
    l = [int(line.strip()) for line in open(filePath + 'split/labels_test_'+str(cv)+'.txt')]
    tel, vl = splitDataSet(l, False)

    tr_bp = numpy.loadtxt(filePath + 'split/BP_train_'+str(cv)+'.txt', delimiter=',')
    bp = numpy.loadtxt(filePath + 'split/BP_test_'+str(cv)+'.txt',delimiter=',')
    te_bp, v_bp = splitDataSet(bp, True)

    tr_cc = numpy.loadtxt(filePath + 'split/CC_train_'+str(cv)+'.txt',delimiter=',')
    cc = numpy.loadtxt(filePath + 'split/CC_test_'+str(cv)+'.txt',delimiter=',')
    te_cc, v_cc = splitDataSet(cc, True)

    tr_mf = numpy.loadtxt(filePath + 'split/MF_train_'+str(cv)+'.txt',delimiter=',')
    mf = numpy.loadtxt(filePath + 'split/MF_test_'+str(cv)+'.txt',delimiter=',')
    te_mf, v_mf = splitDataSet(mf, True)

    if not weight:
        [train_set_x1, train_set_x2, train_bp, train_cc, train_mf], train_set_y = shared_dataset([trda, trdb, tr_bp, tr_cc, tr_mf], trl)
        [test_set_x1, test_set_x2, test_bp, test_cc, test_mf], test_set_y = shared_dataset([teda, tedb, te_bp, te_cc, te_mf], tel)
        [valid_set_x1, valid_set_x2, valid_bp, valid_cc, valid_mf], valid_set_y = shared_dataset([vda, vdb, v_bp, v_cc, v_mf], vl)

        rval = [(train_set_x1, train_set_x2, train_set_y, train_bp, train_cc, train_mf), (test_set_x1, test_set_x2, test_set_y, test_bp, test_cc, test_mf),
                (valid_set_x1, valid_set_x2, valid_set_y, valid_bp, valid_cc, valid_mf)]
        return rval
    else:
        trw_bp = numpy.loadtxt(filePath + 'split/BP_trainWT_'+str(cv)+'.txt', delimiter=',')
        w_bp = numpy.loadtxt(filePath + 'split/BP_testWT_'+str(cv)+'.txt', delimiter=',')
        tew_bp, vw_bp = splitDataSet(w_bp, False)
        tra_bp = numpy.loadtxt(filePath + 'split/BP_trainAVL_'+str(cv)+'.txt', delimiter=',')
        a_bp = numpy.loadtxt(filePath + 'split/BP_testAVL_'+str(cv)+'.txt', delimiter=',')
        tea_bp, va_bp = splitDataSet(a_bp, False)

        trw_cc = numpy.loadtxt(filePath + 'split/CC_trainWT_'+str(cv)+'.txt', delimiter=',')
        w_cc = numpy.loadtxt(filePath + 'split/CC_testWT_'+str(cv)+'.txt', delimiter=',')
        tew_cc, vw_cc = splitDataSet(w_cc, False)
        tra_cc = numpy.loadtxt(filePath + 'split/CC_trainAVL_'+str(cv)+'.txt', delimiter=',')
        a_cc = numpy.loadtxt(filePath + 'split/CC_testAVL_'+str(cv)+'.txt', delimiter=',')
        tea_cc, va_cc = splitDataSet(a_cc, False)

        trw_mf = numpy.loadtxt(filePath + 'split/MF_trainWT_'+str(cv)+'.txt', delimiter=',')
        w_mf = numpy.loadtxt(filePath + 'split/MF_testWT_'+str(cv)+'.txt', delimiter=',')
        tew_mf, vw_mf = splitDataSet(w_mf, False)
        tra_mf = numpy.loadtxt(filePath + 'split/MF_trainAVL_'+str(cv)+'.txt', delimiter=',')
        a_mf = numpy.loadtxt(filePath + 'split/MF_testAVL_'+str(cv)+'.txt', delimiter=',')
        tea_mf, va_mf = splitDataSet(a_mf, False)

        [train_set_x1, train_set_x2, train_bp, train_cc, train_mf, train_w_bp, train_a_bp, train_w_cc, train_a_cc, train_w_mf, train_a_mf], train_set_y = \
            shared_dataset([trda, trdb, tr_bp, tr_cc, tr_mf, trw_bp, tra_bp, trw_cc, tra_cc, trw_mf, tra_mf], trl)
        [test_set_x1, test_set_x2, test_bp, test_cc, test_mf, test_w_bp, test_a_bp, test_w_cc, test_a_cc, test_w_mf, test_a_mf], test_set_y =\
            shared_dataset([teda, tedb, te_bp, te_cc, te_mf, tew_bp, tea_bp, tew_cc, tea_cc, tew_mf, tea_mf], tel)
        [valid_set_x1, valid_set_x2, valid_bp, valid_cc, valid_mf, valid_w_bp, valid_a_bp, valid_w_cc, valid_a_cc, valid_w_mf, valid_a_mf], valid_set_y =\
            shared_dataset([vda, vdb, v_bp, v_cc, v_mf, vw_bp, va_bp, vw_cc, va_cc, vw_mf, va_mf], tel)

        rval = [(train_set_x1, train_set_x2, train_set_y, train_bp, train_cc, train_mf,
                 train_w_bp, train_a_bp, train_w_cc, train_a_cc, train_w_mf, train_a_mf),
                (test_set_x1, test_set_x2, test_set_y, test_bp, test_cc, test_mf,
                 test_w_bp, test_a_bp, test_w_cc, test_a_cc, test_w_mf, test_a_mf),
                (valid_set_x1, valid_set_x2, valid_set_y, valid_bp, valid_cc, valid_mf,
                 valid_w_bp, valid_a_bp, valid_w_cc, valid_a_cc, valid_w_mf, valid_a_mf)]
        return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values


if __name__ == '__main__':
    sgd_optimization_mnist()

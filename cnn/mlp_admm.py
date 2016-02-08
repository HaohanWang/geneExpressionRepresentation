import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data, LinearRegression
from cnn import LeNetConvPoolLayer
from optimizers import Optimizer

from matplotlib import pyplot as plt


# TODO: EVALUATE THE WEIGHT OF EACH INPUT DATA, TRY WEIGHTED LEARNING


def tanh(x):
    return T.tanh(x)


def rectifier(x):
    return T.maximum(0., x)


def linear(x):
    return x


def sigmoid(x):
    return T.nnet.sigmoid(x)


def params_shape_like(params):
    l = []
    for k in params:
        l.append(numpy.zeros_like(k.get_value(True)))
    return l


def update_params(param_g, param_s):
    # return (param_g + param_s) / 2
    # zeroIndex = numpy.where(param_g*param_s==0)
    # ones = numpy.ones_like(param_g)
    # ones[zeroIndex] = 0
    mean = (param_g + param_s) / 2
    # r = mean*ones
    return mean

def normalizedVector(vec =[]):
    vec = numpy.array(vec)
    maxi = numpy.max(vec)
    mini = numpy.min(vec)
    return (vec-mini)/(maxi-mini)


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

        h_output = [out2_a, out2_b]
        shp = out2_a.shape
        h_output = T.reshape(h_output, [shp[0], shp[1] * 2])

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


def test_mlp(learning_rate=0.1, L1_reg=(), L2_reg=(), D_reg=1.0, BP_reg=0.0, CC_reg=0.0, MF_reg=0.0, rho=0,
             n_epochs=1000, batch_size=1000, cv=1):
    datasets = load_data(cv)

    train_set_x1, train_set_x2, train_set_y, train_bp, train_cc, train_mf = datasets[0]
    valid_set_x1, valid_set_x2, valid_set_y, valid_bp, valid_cc, valid_mf = datasets[1]

    n_train_batches = train_set_x1.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x1.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    index = T.lscalar()
    x1 = T.matrix('x1')
    x2 = T.matrix('x2')
    y = T.ivector('y')
    y_bp = T.matrix('bp')
    y_cc = T.matrix('cc')
    y_mf = T.matrix('mf')

    hw1 = T.matrix('hw1')
    hb1 = T.dvector('hb1')
    hw2 = T.matrix('hw2')
    hb2 = T.dvector('hb2')
    lw = T.matrix('lw')
    lb = T.dvector('hb')
    lwbp = T.matrix('lwbp')
    lbbp = T.dvector('hbbp')
    lwcc = T.matrix('lwcc')
    lbcc = T.dvector('hbcc')
    lwmf = T.matrix('lwmf')
    lbmf = T.dvector('hbmf')

    hw1_s = T.matrix('hw1_s')
    hb1_s = T.dvector('hb1_s')
    hw2_s = T.matrix('hw2_s')
    hb2_s = T.dvector('hb2_s')
    lw_s = T.matrix('lw_s')
    lb_s = T.dvector('hb_s')
    lwbp_s = T.matrix('lwbp_s')
    lbbp_s = T.dvector('hbbp_s')
    lwcc_s = T.matrix('lwcc_s')
    lbcc_s = T.dvector('hbcc_s')
    lwmf_s = T.matrix('lwmf_s')
    lbmf_s = T.dvector('hbmf_s')

    hw1_g = T.matrix('hw1_g')
    hb1_g = T.dvector('hb1_g')
    hw2_g = T.matrix('hw2_g')
    hb2_g = T.dvector('hb2_g')
    lw_g = T.matrix('lw_g')
    lb_g = T.dvector('hb_g')
    lwbp_g = T.matrix('lwbp_g')
    lbbp_g = T.dvector('hbbp_g')
    lwcc_g = T.matrix('lwcc_g')
    lbcc_g = T.dvector('hbcc_g')
    lwmf_g = T.matrix('lwmf_g')
    lbmf_g = T.dvector('hbmf_g')

    mu = 0

    rng = numpy.random.RandomState(1234)

    activation = tanh

    classifier = MLP(
            rng=rng,
            input1=x1,
            input2=x2,
            configs=(158, 100, 50),
            n_out=2,
            batch_size=batch_size,
            activation=activation
    )

    classifier_semantic = MLP(
            rng=rng,
            input1=x1,
            input2=x2,
            configs=(158, 100, 50),
            n_out=2,
            batch_size=batch_size,
            activation=activation
    )

    classifier_graphic = MLP(
            rng=rng,
            input1=x1,
            input2=x2,
            configs=(158, 100, 50),
            n_out=2,
            batch_size=batch_size,
            activation=activation
    )

    params_semantic = params_shape_like(classifier_semantic.params)
    params_graphic = params_shape_like(classifier_graphic.params[:4])
    params_update = params_shape_like(classifier.params)

    cost = (
        classifier.negative_log_likelihood(y)
        + classifier.L1_reg_all(L1_reg)
        + classifier.L2_reg_all(L2_reg)
        + D_reg * classifier.distance(y) / batch_size
        + BP_reg * classifier.mse_bp(y_bp)
        + CC_reg * classifier.mse_cc(y_cc)
        + MF_reg * classifier.mse_mf(y_mf)
    )

    cost_semantic = (
        classifier_semantic.negative_log_likelihood(y)
        + BP_reg * classifier_semantic.mse_bp(y_bp)
        + CC_reg * classifier_semantic.mse_cc(y_cc)
        + MF_reg * classifier_semantic.mse_mf(y_mf)
        + classifier_semantic.L1_reg_all(L1_reg)
        + classifier_semantic.L2_reg_all(L2_reg)
        + classifier_semantic.augment([hw1_g, hb1_g, hw2_g, hb2_g], mu, rho)
    )

    cost_graphic = (
        classifier_graphic.distance(y) / batch_size
        + classifier_graphic.L1_reg_lower(L1_reg[:4])
        + classifier_graphic.L2_reg_lower(L2_reg[:4])
        + classifier_graphic.augment([hw1_s, hb1_s, hw2_s, hb2_s], mu, rho)
    )

    opt = Optimizer()
    optFunc = opt.adam

    updates_semantic = optFunc(cost_semantic, classifier_semantic.params, learning_rate)
    updates_graphic = optFunc(cost_graphic, classifier_graphic.params[:4], learning_rate)

    # train_model = theano.function(
    #         inputs=[index],
    #         outputs=cost,
    #         updates=updates,
    #         givens={
    #             x1: train_set_x1[index * batch_size: (index + 1) * batch_size],
    #             x2: train_set_x2[index * batch_size: (index + 1) * batch_size],
    #             y: train_set_y[index * batch_size: (index + 1) * batch_size],
    #             y_bp: train_bp[index * batch_size:(index + 1) * batch_size],
    #             y_mf: train_mf[index * batch_size:(index + 1) * batch_size],
    #             y_cc: train_cc[index * batch_size:(index + 1) * batch_size],
    #         }
    # )

    train_model_semantic = theano.function(
            inputs=[index, hw1_g, hb1_g, hw2_g, hb2_g],
            outputs=cost_semantic,
            updates=updates_semantic,
            givens={
                x1: train_set_x1[index * batch_size: (index + 1) * batch_size],
                x2: train_set_x2[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                y_bp: train_bp[index * batch_size:(index + 1) * batch_size],
                y_mf: train_mf[index * batch_size:(index + 1) * batch_size],
                y_cc: train_cc[index * batch_size:(index + 1) * batch_size],
            }
    )

    train_model_graphic = theano.function(
            inputs=[index, hw1_s, hb1_s, hw2_s, hb2_s],
            outputs=cost_graphic,
            updates=updates_graphic,
            givens={
                x1: train_set_x1[index * batch_size: (index + 1) * batch_size],
                x2: train_set_x2[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
    )

    update_model = theano.function(
            inputs=[hw1, hb1, hw2, hb2, lw, lb, lwbp, lbbp, lwcc, lbcc, lwmf, lbmf],
            updates=[(param, uparam)
                     for param, uparam in
                     zip(classifier.params, [hw1, hb1, hw2, hb2, lw, lb, lwbp, lbbp, lwcc, lbcc, lwmf, lbmf])
                     ]
    )

    validate_model = theano.function(
            inputs=[index],
            outputs=(classifier.errors(y), classifier.distance(y) / batch_size,
                     classifier.mse_bp(y_bp), classifier.mse_cc(y_cc), classifier.mse_mf(y_mf)),
            givens={
                x1: valid_set_x1[index * batch_size:(index + 1) * batch_size],
                x2: valid_set_x2[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size],
                y_bp: valid_bp[index * batch_size:(index + 1) * batch_size],
                y_mf: valid_mf[index * batch_size:(index + 1) * batch_size],
                y_cc: valid_cc[index * batch_size:(index + 1) * batch_size]
            }
    )

    print '... training'

    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 5  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_validation_distance = numpy.inf
    best_validation_mse_bp = numpy.inf
    best_validation_mse_mf = numpy.inf
    best_validation_mse_cc = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    minibatch_avg_cost = 0

    minibatch_avg_cost_semantic = 0
    minibatch_avg_cost_graphic = 0

    vloss = []
    vdist = []
    vbp = []
    vmf = []
    vcc = []

    # best_script = open('mlp.py')
    while (epoch < n_epochs) and (not done_looping):
        # print classifier.CovPol.params[-1].get_value(True)
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            # minibatch_avg_cost = train_model(minibatch_index)

            minibatch_avg_cost_semantic = train_model_semantic(minibatch_index, params_graphic[0], params_graphic[1],
                                                               params_graphic[2], params_graphic[3])
            minibatch_avg_cost_graphic = train_model_graphic(minibatch_index, params_semantic[0], params_semantic[1],
                                                             params_semantic[2], params_semantic[3])

            for p_index in range(len(classifier.params)):
                params_semantic[p_index] = classifier_semantic.params[p_index].get_value(True)
                if p_index < 4:
                    params_graphic[p_index] = classifier_graphic.params[p_index].get_value(True)
                    params_update[p_index] = update_params(param_g=params_graphic[p_index], param_s=params_semantic[p_index])
                else:
                    params_update[p_index] = params_semantic[p_index]

            update_model(params_update[0], params_update[1], params_update[2], params_update[3], params_update[4],
                         params_update[5], params_update[6], params_update[7], params_update[8], params_update[9],
                         params_update[10], params_update[11])

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                validation_loss, validation_distance, mse_bp, mse_cc, mse_mf = zip(*validation_losses)
                this_validation_loss = numpy.mean(validation_loss)
                this_validation_distance = numpy.mean(validation_distance)

                this_validation_mse_bp = numpy.mean(mse_bp)
                this_validation_mse_cc = numpy.mean(mse_cc)
                this_validation_mse_mf = numpy.mean(mse_mf)

                print 'epoch', epoch, 'minibatch', str(minibatch_index + 1) + '/' + str(n_train_batches),
                print 'validation error', this_validation_loss * 100, '%'
                print 'validation distance', this_validation_distance
                print 'validation mse bp, cc, mf', this_validation_mse_bp, this_validation_mse_cc, this_validation_mse_mf

                vloss.append(this_validation_loss)
                vdist.append(this_validation_distance)
                vbp.append(this_validation_mse_bp)
                vmf.append(this_validation_mse_mf)
                vcc.append(this_validation_mse_cc)

                if this_validation_loss < best_validation_loss:
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_score = best_validation_loss

                    print '----BEST Validated MODEL here', test_score * 100, '%----'

                if this_validation_distance < best_validation_distance:
                    if (this_validation_distance < best_validation_distance * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_distance = this_validation_distance

                if this_validation_mse_bp < best_validation_mse_bp:
                    if (this_validation_mse_bp < best_validation_mse_bp * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_mse_bp = this_validation_mse_bp

                if this_validation_mse_cc < best_validation_mse_cc:
                    if (this_validation_mse_cc < best_validation_mse_cc * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_mse_cc = this_validation_mse_cc

                if this_validation_mse_mf < best_validation_mse_mf:
                    if (this_validation_mse_mf < best_validation_mse_mf * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_mse_mf = this_validation_mse_mf

            if patience <= iter:
                done_looping = True
                break
        print minibatch_avg_cost_semantic, minibatch_avg_cost_graphic
        # print classifier.params[1].get_value(True)

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    vloss = normalizedVector(vloss)
    vdist = normalizedVector(vdist)
    vbp = normalizedVector(vbp)
    vmf = normalizedVector(vmf)
    vcc = normalizedVector(vcc)

    x = vloss.shape[0]
    plt.plot(x, vloss, label='loss')
    plt.plot(x, vdist, label='dist')
    plt.plot(x, vbp, label='bp')
    plt.plot(x, vmf, label='mf')
    plt.plot(x, vcc, label='cc')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    import sys
    # args = sys.argv
    # lr = float(args[1])
    # batch_size = int(args[2])
    lr = 0.001
    batch_size = 1000
    l1 = (5e-5,5e-5,5e-5,5e-5,0,0,0,0,0,0,0,0)
    l2 = (0,0,0,0,0,0,0,0,0,0,0,0)
    dr = 1e-3
    bp_reg = 1e-3
    cc_reg = 1e-3
    mf_reg = 1e-3
    test_mlp(cv=1, learning_rate=lr, L1_reg=l1, L2_reg=l2, D_reg=dr, BP_reg=bp_reg, CC_reg=cc_reg, MF_reg=mf_reg,
             batch_size=batch_size)
    print l1, l2, dr, bp_reg, cc_reg, mf_reg

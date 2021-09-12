import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import cifar10_data
from checkpoints import save_weights,load_weights

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)     #random seed for theano operation
parser.add_argument('--seed_data', type=int, default=1)  #random seed for picking labeled data
parser.add_argument('--count', type=int, default=400)   #how much data one class
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--base_RF_loss_weight', type=float, default=0.01)  #weight for base random field loss, i.e. f-E[f]
parser.add_argument('--lrd', type=float, default=1e-3)
parser.add_argument('--lrg', type=float, default=1e-3)
parser.add_argument('--potential_control_weight', default=1e-3 ,type=float)    #weight for confidence loss
parser.add_argument('--beta', default=0.5 ,type=float)   #beta for SGHMC
parser.add_argument('--gradient_coefficient', default=0.003,type=float)  #coefficient for gradient term of SGLD/SGHMC
parser.add_argument('--noise_coefficient', default=0,type=float)   #coefficient for noise term of SGLD/SGHMC
parser.add_argument('--L', default=10 ,type=int)   #revision steps
parser.add_argument('--max_e', default=600 ,type=int)   #max number of epochs
parser.add_argument('--revison_method', default='revision_x_sghmc' ,type=str)   #revision method
parser.add_argument('--load', default='' ,type=str)    #file name to load trained model
parser.add_argument('--data_dir', type=str, default='../data/cifar-10-python/')   #data folder to load
args = parser.parse_args()
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR data
def rescale(mat):
    return np.transpose(np.cast[th.config.floatX]((-127.5 + mat)/127.5),(3,2,0,1))

trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
testx, testy = cifar10_data.load(args.data_dir, subset='test')
trainx_unl = np.array(trainx).copy()
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(np.ceil(float(testx.shape[0])/args.batch_size))

# specify generator
h = T.matrix()
gen_layers = [ll.InputLayer(shape=(None, 100))]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu,name='g_1' ), g=None,name='g_b1'))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1,512,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu,name='g_2'), g=None,name='g_b2')) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu,name='g_3'), g=None,name='g_b3')) # 8 -> 16
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (None,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh,name='g_4'), train_g=True, init_stdv=0.1,name='g_w4')) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1],h,deterministic=False)

# specify random field
layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
layers.append(nn.weight_norm(ll.Conv2DLayer(layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_1'),name='d_w1'))
layers.append(nn.weight_norm(ll.Conv2DLayer(layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_2'),name='d_w2'))
layers.append(nn.weight_norm(ll.Conv2DLayer(layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_3'),name='d_w3'))
layers.append(ll.MaxPool2DLayer(layers[-1],(2,2)))
layers.append(ll.DropoutLayer(layers[-1], p=0.5))
layers.append(nn.weight_norm(ll.Conv2DLayer(layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_4'),name='d_w4'))
layers.append(nn.weight_norm(ll.Conv2DLayer(layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_5'),name='d_w5'))
layers.append(nn.weight_norm(ll.Conv2DLayer(layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_6'),name='d_w6'))
layers.append(ll.MaxPool2DLayer(layers[-1],(2,2)))
layers.append(ll.DropoutLayer(layers[-1], p=0.5))
layers.append(nn.weight_norm(ll.Conv2DLayer(layers[-1],512, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_7'),name='d_w7'))
layers.append(nn.weight_norm(ll.NINLayer(layers[-1], num_units=256, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_8'),name='d_w8'))
layers.append(nn.weight_norm(ll.NINLayer(layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu,name='d_9'),name='d_w9'))
layers.append(ll.GlobalPoolLayer(layers[-1]))
layers.append(nn.weight_norm(ll.DenseLayer(layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None,name='d_10'), train_g=True, init_stdv=0.1,name='d_w10'))

# costs

#revision method
if args.revison_method=='revision_x_sgld':    #only x will be revised, SGLD
    x_revised = gen_dat
    gradient_coefficient = T.scalar()
    noise_coefficient = T.scalar()
    for i in range(args.L):
        loss_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_revision, [x_revised])[0]
        x_revised = x_revised + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))

    revision = th.function(inputs=[h, gradient_coefficient, noise_coefficient], outputs=x_revised)

elif args.revison_method=='revision_x_sghmc':  #only x will be revised, SGHMC
    x_revised = gen_dat
    gradient_coefficient = T.scalar()
    beta = T.scalar()
    noise_coefficient = T.scalar()
    v_x = 0.
    for i in range(args.L):
        loss_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_revision, [x_revised])[0]
        v_x = beta * v_x + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))
        x_revised = x_revised + v_x

    revision = th.function(inputs=[h, beta, gradient_coefficient, noise_coefficient], outputs=x_revised)
elif args.revison_method=='revision_joint_sgld':  #x and h will be revised jointly, SGLD
    x_revised = gen_dat
    h_revised = h
    gradient_coefficient = T.scalar()
    noise_coefficient = T.scalar()
    for i in range(args.L):

        loss_x_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_x_revision, [x_revised])[0]
        x_revised = x_revised + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))
        if i==0:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat)) + T.sum(T.square(h))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h])[0]
            h_revised= h - gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
        else:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat_h_revised))+ T.sum(T.square(h_revised))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h_revised])[0]
            h_revised = h_revised - gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
        gen_dat_h_revised=ll.get_output(gen_layers[-1],h_revised, deterministic=False)

    revision = th.function(inputs=[h, gradient_coefficient, noise_coefficient], outputs=[x_revised,h_revised])
elif args.revison_method=='revision_joint_sghmc':   #x and h will be revised jointly, SGHMC
    x_revised = gen_dat
    h_revised = h
    beta=T.scalar()
    gradient_coefficient = T.scalar()
    noise_coefficient = T.scalar()
    v_x=0.
    for i in range(args.L):

        loss_x_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_x_revision, [x_revised])[0]
        v_x=v_x*beta + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))
        x_revised = x_revised + v_x

        if i==0:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat))+ T.sum(T.square(h))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h])[0]
            v_h= gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
            h_revised= h - v_h

        else:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat_h_revised))+ T.sum(T.square(h_revised))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h_revised])[0]
            v_h=v_h*beta+gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
            h_revised = h_revised - v_h
            gen_dat_h_revised=ll.get_output(gen_layers[-1],h_revised, deterministic=False)

    revision = th.function(inputs=[h, beta,gradient_coefficient, noise_coefficient], outputs=[x_revised,h_revised])

potential_control_weight = T.scalar()
base_RF_loss_weight=T.scalar()
x_revised = T.tensor4()
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()

temp = ll.get_output(gen_layers[-1],h, deterministic=False, init=True)
temp = ll.get_output(layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(layers[-1], x_unl, deterministic=False)
output_before_softmax_sam = ll.get_output(layers[-1], x_revised, deterministic=False)

logit_lab = output_before_softmax_lab[T.arange(T.shape(x_lab)[0]),labels]

u_lab = T.mean(nn.log_sum_exp(output_before_softmax_lab))
u_unl = T.mean(nn.log_sum_exp(output_before_softmax_unl))
u_revised = T.mean(nn.log_sum_exp(output_before_softmax_sam))
#cross entropy loss of labeled data
loss_lab = -T.mean(logit_lab) + u_lab


loss_unl = (u_revised-u_unl)*base_RF_loss_weight+ T.mean(T.square(nn.log_sum_exp(output_before_softmax_unl)))*potential_control_weight
train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

loss_RF=loss_lab+loss_unl
# test error
output_before_softmax = ll.get_output(layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))


# Theano functions for training the random field
lr = T.scalar()
RF_params = ll.get_all_params(layers, trainable=True)
RF_param_updates = lasagne.updates.rmsprop(loss_RF, RF_params, learning_rate=lr)
train_RF = th.function(inputs=[x_revised,x_lab,labels,x_unl,lr,base_RF_loss_weight,potential_control_weight], outputs=[loss_lab, loss_unl, train_err], updates=RF_param_updates)
#weight norm initalization
init_param = th.function(inputs=[h,x_lab], outputs=None, updates=init_updates)
#predition on test data
output_before_softmax = ll.get_output(layers[-1], x_lab, deterministic=True)
test_batch = th.function(inputs=[x_lab], outputs=output_before_softmax)

#loss on generator
loss_G = T.sum(T.square(x_revised - gen_dat))
# Theano functions for training the generator
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates =lasagne.updates.rmsprop(loss_G, gen_params, learning_rate=lr)
train_G = th.function(inputs=[h,x_revised,lr], outputs=None, updates=gen_param_updates)


# select labeled data
rng_data = np.random.RandomState(args.seed_data)
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

# //////////// perform training //////////////
lr_D=args.lrd
lr_G=args.lrg
beta=args.beta
gradient_coefficient=args.gradient_coefficient
noise_coefficient=args.noise_coefficient
base_RF_loss_weight = args.base_RF_loss_weight
potential_control_weight=args.potential_control_weight
acc_all=[]
best_acc=1
for epoch in range(args.max_e):
    begin = time.time()
    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    if epoch==0:
        init_param(np.cast[th.config.floatX](np.random.uniform(size=(500,100))),trainx[:500]) # data based initialization
        if args.load:
            load_weights('cifar_model/cifar_jrf_' + args.load + '.npy', layers + gen_layers)

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.

    for t in range(nr_batches_train):
        h = np.cast[th.config.floatX](rng.uniform(size=(args.batch_size, 100)))
        if args.revison_method=='revision_x_sgld':
            x_revised = revision(h, gradient_coefficient, noise_coefficient)
        elif args.revison_method=='revision_x_sghmc':
            x_revised= revision(h, beta, gradient_coefficient, noise_coefficient)
        elif args.revison_method == 'revision_joint_sgld':
            x_revised,h = revision(h, gradient_coefficient, noise_coefficient)
        elif args.revison_method == 'revision_joint_sghmc':
            x_revised,h = revision(h, beta, gradient_coefficient, noise_coefficient)
        ran_from = t * args.batch_size
        ran_to = (t + 1) * args.batch_size
        # updata random field
        lo_lab, lo_unl, tr_er = train_RF(x_revised, trainx[ran_from:ran_to], trainy[ran_from:ran_to],
                                         trainx_unl[ran_from:ran_to], lr_D, base_RF_loss_weight,
                                         potential_control_weight)
        loss_lab += lo_lab
        loss_unl += lo_unl
        train_err += tr_er
        # updata generator
        train_G(h, x_revised, lr_G)
    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    # test
    test_pred = np.zeros((len(testy), 10), dtype=th.config.floatX)
    for t in range(nr_batches_test):
        last_ind = np.minimum((t + 1) * args.batch_size, len(testy))
        first_ind = last_ind - args.batch_size
        test_pred[first_ind:last_ind] = test_batch(testx[first_ind:last_ind])
    test_err = np.mean(np.argmax(test_pred, axis=1) != testy)
    print(
        "epoch %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f, best_err = %.4f" % (
            epoch + 1, time.time() - begin, loss_lab, loss_unl, train_err, test_err, best_acc))
    sys.stdout.flush()
    acc_all.append(test_err)

    if acc_all[-1] < best_acc:
        best_acc = acc_all[-1]
    if (epoch + 1) % 50 == 0:
        import os
        if not os.path.exists('cifar_model'):
            os.mkdir('cifar_model')
        params = ll.get_all_params(layers + gen_layers)
        save_weights('cifar_model/semi_nrf_data%d_ep%d.npy' % (args.seed_data, epoch + 1), params)









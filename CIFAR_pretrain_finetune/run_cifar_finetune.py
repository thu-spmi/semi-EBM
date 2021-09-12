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
parser.add_argument('--data_dir', type=str, default='data/cifar-10-python/')   #data folder to load
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


labels = T.ivector()
x_lab = T.tensor4()

temp = ll.get_output(layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(layers[-1], x_lab, deterministic=False)


logit_lab = output_before_softmax_lab[T.arange(T.shape(x_lab)[0]),labels]

u_lab = T.mean(nn.log_sum_exp(output_before_softmax_lab))

#cross entropy loss of labeled data
loss_lab = -T.mean(logit_lab) + u_lab

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))


# Theano functions for training the random field
lr = T.scalar()
RF_params = ll.get_all_params(layers, trainable=True)
RF_param_updates = lasagne.updates.rmsprop(loss_lab, RF_params, learning_rate=lr)
train_RF = th.function(inputs=[x_lab,labels,lr], outputs=[loss_lab, train_err], updates=RF_param_updates)
#weight norm initalization
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
#predition on test data
output_before_softmax = ll.get_output(layers[-1], x_lab, deterministic=True)
test_batch = th.function(inputs=[x_lab], outputs=output_before_softmax)


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
    if epoch==0:
        init_param(trainx[:500]) # data based initialization
        if args.load:
            load_weights('cifar_model/%s.npy'%args.load, layers)
            print('loaded!')
        # load_weights('cifar_model/pretrain_nrf_ep150.npy' , layers)

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.

    for t in range(nr_batches_train):
        ran_from = t * args.batch_size
        ran_to = (t + 1) * args.batch_size
        # updata random field
        lo_lab, tr_er = train_RF( trainx[ran_from:ran_to], trainy[ran_from:ran_to], lr_D)
        loss_lab += lo_lab
        train_err += tr_er
        # updata generator
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
        "epoch %d, time = %ds, loss_lab = %.4f, train err = %.4f, test err = %.4f, best_err = %.4f" % (
            epoch + 1, time.time() - begin, loss_lab, train_err, test_err, best_acc))
    sys.stdout.flush()
    acc_all.append(test_err)

    if acc_all[-1] < best_acc:
        best_acc = acc_all[-1]
    if (epoch + 1) % 50 == 0:
        import os
        if not os.path.exists('cifar_model'):
            os.mkdir('cifar_model')
        params = ll.get_all_params(layers)
        save_weights('cifar_model/finetune_nrf_data%d_ep%d.npy' % (args.seed_data, epoch + 1), params)









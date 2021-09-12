import pickle
import gzip, logging, os
import os.path as osp
import numpy as np


def convert2dict(params):
    names = [par.name for par in params]
    assert len(names) == len(set(names))

    param_dict = {par.name: par.get_value() for par in params}
    return param_dict


def save_weights(fname, params):
    param_dict = convert2dict(params)
    logging.info('saving {} parameters to {}'.format(len(params), fname))
    filename, ext = osp.splitext(fname)
    if ext == '.npy':
        np.save(filename + '.npy', param_dict)
    else:
        f = gzip.open(fname, 'wb')
        pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def load_dict(fname):
    logging.info("Loading weights from {}".format(fname))
    filename, ext = os.path.splitext(fname)
    if ext == '.npy':
        params_load = np.load(fname,allow_pickle=True).item()
    else:
        f = gzip.open(fname, 'r')
        params_load = pickle.load(f)
        f.close()
    if type(params_load) is dict:
        param_dict = params_load
    else:
        param_dict = convert2dict(params_load)
    return param_dict

def load_weights_trainable(fname, l_out):
    import lasagne
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    names = [par.name for par in params]
    assert len(names) == len(set(names))

    if type(fname) is list:
        param_dict = {}
        for name in fname:
            t_load = load_dict(name)
            param_dict.update(t_load)
    else:
        param_dict = load_dict(fname)

    for param in params:
        if param.name in param_dict:
            stored_shape = np.asarray(param_dict[param.name].shape)
            param_shape = np.asarray(param.get_value().shape)
            if not np.all(stored_shape == param_shape):
                warn_msg = 'shape mismatch:'
                warn_msg += '{} stored:{} new:{}'.format(
                    param.name, stored_shape, param_shape)
                warn_msg += ', skipping'
                logging.warn(warn_msg)
            else:
                param.set_value(param_dict[param.name])
        else:
            logging.warn('unable to load parameter {} from {}: No such variable.'
                         .format(param.name, fname))



def load_weights(fname, l_out):
    import lasagne
    params = lasagne.layers.get_all_params(l_out)
    names = [par.name for par in params]
    assert len(names) == len(set(names))

    if type(fname) is list:
        param_dict = {}
        for name in fname:
            t_load = load_dict(name)
            param_dict.update(t_load)
    else:
        param_dict = load_dict(fname)
    assign_weights(params, param_dict)

def assign_weights(params, param_dict):
    for param in params:
        if param.name in param_dict:
            stored_shape = np.asarray(param_dict[param.name].shape)
            param_shape = np.asarray(param.get_value().shape)
            if not np.all(stored_shape == param_shape):
                warn_msg = 'shape mismatch:'
                warn_msg += '{} stored:{} new:{}'.format(
                    param.name, stored_shape, param_shape)
                warn_msg += ', skipping'
                logging.warn(warn_msg)
            else:
                param.set_value(param_dict[param.name])
        else:
            logging.warn('Unable to load parameter {}: No such variable.'
                         .format(param.name))


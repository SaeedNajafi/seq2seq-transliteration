import numpy as np
from numpy import *
import utils as ut

def load(config, mode):
    """
    Creates random vectors for source and target characters.
    Loads train/dev/test data.
    """

    print "INFO: creating random character vectors!"
    s_chars = []
    f = open(config.source_alphabet, 'r')
    lines = f.readlines()
    for line in lines:
        s_chars.append(line.decode('utf8').strip())
    s_chars.append("_")

    config.s_alphabet_size = len(s_chars)

    t_chars = []
    f = open(config.target_alphabet, 'r')
    lines = f.readlines()
    for line in lines:
        t_chars.append(line.decode('utf8').strip())
    t_chars.append("_")

    config.t_alphabet_size = len(t_chars)

    s_num_to_char = dict(enumerate(s_chars))
    s_char_to_num = {v:k for k,v in s_num_to_char.iteritems()}

    t_num_to_char = dict(enumerate(t_chars))
    t_char_to_num = {v:k for k,v in t_num_to_char.iteritems()}

    train_data = None
    dev_data = None
    test_data = None

    if mode == 'train':
        #Loads the training set
        print "INFO: Reading training data!"
        train_data = load_dataset(
                            config.train_set,
                            s_chars,
                            t_chars,
                            s_char_to_num,
                            t_char_to_num,
                            config.max_length,
                            mode
                            )

        #Loads the dev set
        print "INFO: Reading dev data!"
        dev_data = load_dataset(
                            config.dev_set,
                            s_chars,
                            t_chars,
                            s_char_to_num,
                            t_char_to_num,
                            config.max_length,
                            mode
                            )

    elif mode == 'test':
        #Loads the test set
        print "INFO: Reading test data!"
        test_data = load_dataset(
                            config.test_set,
                            s_chars,
                            t_chars,
                            s_char_to_num,
                            t_char_to_num,
                            config.max_length,
                            mode
                            )

    ret = {}
    ret['s_num_to_char'] = s_num_to_char
    ret['t_num_to_char'] = t_num_to_char
    ret['train_data'] = train_data
    ret['dev_data'] = dev_data
    ret['test_data'] = test_data

    return ret

def load_dataset(
                fname,
                s_chars,
                t_chars,
                s_char_to_num,
                t_char_to_num,
                max_length,
                mode
                ):

    X_data = []
    X_length = []
    X_mask = []

    Y_data = []
    Y_length = []
    Y_mask = []
    with open(fname) as f:
        for line in f:
            XY = line.decode('utf-8').strip().split("\t")
            if len(XY) > 0:
                X = map_char(XY[0], s_chars, s_char_to_num)
                ln = len(X)
                X_length.append(ln)
                mask_list = [1.0] * ln
                if ln < max_length:
                    while ln < max_length:
                        X.append(0)
                        mask_list.append(0.0)
                        ln += 1
                else:
                    X = X[0:max_length]
                    mask_list = mask_list[0:max_length]

                X_data.append(X)
                X_mask.append(mask_list)

                if mode=='train':
                    Y = map_char(XY[1], t_chars, t_char_to_num)
                    ln = len(Y)
                    Y_length.append(ln)
                    mask_list = [1.0] * ln
                    if ln < max_length:
                        while ln < max_length:
                            Y.append(0)
                            mask_list.append(0.0)
                            ln += 1
                    else:
                        Y = Y[0:max_length]
                        mask_list = mask_list[0:max_length]

                    Y_data.append(Y)
                    Y_mask.append(mask_list)

                else:
                    print "Wrong input file format!"
                    exit()

    ret = {
            'X': array(X_data),
            'X_length': array(X_length),
            'X_mask': array(X_mask),
            'Y': array(Y_data),
            'Y_length': array(Y_length),
            'Y_mask': array(Y_mask)
            }

    return ret

def map_char(word, charset, map_to_num):
    out = []
    for each in list(word):
        if each in charset:
            out.append(map_to_num[each])
        else:
            print "could not find the char:" + each

    #adding end sign
    out.append(map_to_num['_'])
    return out

def data_iterator(
        X,
        X_length,
        X_mask,
        Y,
        Y_length,
        Y_mask,
        batch_size,
        shuffle
        ):

    total_steps = int(np.ceil(len(X) / float(batch_size)))
    if shuffle:
    	steps = np.random.permutation(total_steps).tolist()
    else:
    	steps = range(total_steps)

    for step in steps:
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        ret_X = X[batch_start:batch_start + batch_size][:]
        ret_X_length = X_length[batch_start:batch_start + batch_size][:]
        ret_X_mask = X_mask[batch_start:batch_start + batch_size][:]

        ret_Y = None
        if np.any(Y):
            ret_Y = Y[batch_start:batch_start + batch_size][:]
            ret_Y_length = Y_length[batch_start:batch_start + batch_size][:]
            ret_Y_mask = Y_mask[batch_start:batch_start + batch_size][:]

        ###
        yield ret_X, ret_X_length, ret_X_mask, ret_Y, ret_Y_length, ret_Y_mask

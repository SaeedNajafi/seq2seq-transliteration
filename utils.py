import numpy as np
import utils as ut

def load(config, mode):
    """
    Creates random vectors for source and target characters.
    Loads train/dev/test data.
    """
    fp = open(config.source_alphabet, 'r')
    s_chars = [line.strip().decode('utf8') for line in fp.readlines()]
    s_chars += config.end_symbol
    config.s_alphabet_size = len(s_chars)
    fp.close()

    fp = open(config.target_alphabet, 'r')
    t_chars = [line.strip().decode('utf8') for line in fp.readlines()]
    t_chars += config.end_symbol
    config.t_alphabet_size = len(t_chars)
    fp.close()

    s_id_to_char = dict(enumerate(s_chars))
    s_char_to_id = {v:k for k,v in s_id_to_char.iteritems()}

    t_id_to_char = dict(enumerate(t_chars))
    t_char_to_id = {v:k for k,v in t_id_to_char.iteritems()}

    train_data = None
    dev_data = None
    test_data = None

    if mode == 'train':
        #Loads the training set
        print "INFO: Reading training data!"
        train_data = load_dataset(
                            config,
                            config.train_set,
                            s_chars,
                            t_chars,
                            s_char_to_id,
                            t_char_to_id,
                            config.max_length,
                            mode
                            )

        #Loads the dev set
        print "INFO: Reading dev data!"
        dev_data = load_dataset(
                            config,
                            config.dev_set,
                            s_chars,
                            t_chars,
                            s_char_to_id,
                            t_char_to_id,
                            config.max_length,
                            mode
                            )

    elif mode == 'test':
        #Loads the test set
        print "INFO: Reading test data!"
        test_data = load_dataset(
                            config,
                            config.test_set,
                            s_chars,
                            t_chars,
                            s_char_to_id,
                            t_char_to_id,
                            config.max_length,
                            mode
                            )

    ret = {}
    ret['s_id_to_char'] = s_id_to_char
    ret['t_id_to_char'] = t_id_to_char
    ret['train_data'] = train_data
    ret['dev_data'] = dev_data
    ret['test_data'] = test_data
    return ret

def load_dataset(
                config,
                fname,
                s_chars,
                t_chars,
                s_char_to_id,
                t_char_to_id,
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
                X = map_char(XY[0], s_chars, s_char_to_id, config)
                ln = len(X)
                X_length.append(ln)
                mask_list = [1.0] * ln
                if ln < max_length:
                    while ln < max_length:
                        X.append(len(s_chars)-1)
                        mask_list.append(0.0)
                        ln += 1
                else:
                    X = X[0:max_length]
                    mask_list = mask_list[0:max_length]

                X_data.append(X)
                X_mask.append(mask_list)

                if mode=='train':
                    Y = map_char(XY[1], t_chars, t_char_to_id, config)
                    ln = len(Y)
                    Y_length.append(ln)
                    mask_list = [1.0] * ln
                    if ln < max_length:
                        while ln < max_length:
                            Y.append(len(t_chars)-1)
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
            'X': np.array(X_data),
            'X_length': np.array(X_length),
            'X_mask': np.array(X_mask),
            'Y': np.array(Y_data),
            'Y_length': np.array(Y_length),
            'Y_mask': np.array(Y_mask)
            }

    return ret

def map_char(word, charset, map_to_id, config):
    try:
        out = [map_to_id[ch] for ch in word if ch in charset]
    except Exception:
        raise KeyError("could not find some chars")

    #adding end sign
    out.append(map_to_id[config.end_symbol])
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
        ret_Y_mask = None
        ret_Y_length = None
        if np.any(Y):
            ret_Y = Y[batch_start:batch_start + batch_size][:]
            ret_Y_length = Y_length[batch_start:batch_start + batch_size][:]
            ret_Y_mask = Y_mask[batch_start:batch_start + batch_size][:]

        ###
        yield ret_X, ret_X_length, ret_X_mask, ret_Y, ret_Y_length, ret_Y_mask

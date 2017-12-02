class Configuration(object):
    """Model hyper parameters and data information"""

    """ Path to different files """
    source_alphabet = './data/EnHe/source.txt'
    target_alphabet = './data/EnHe/target.txt'
    train_set = './data/EnHe/mytrain.txt'
    dev_set = './data/EnHe/mydev.txt'
    test_set = './data/EnHe/mytest.txt'
    end_symbol = '#'

    """ Neural hyper params """
    s_embedding_size = 128
    s_alphabet_size = None
    t_embedding_size = 128
    t_alphabet_size = None
    max_length = 32
    h_units = 256
    batch_size = 4
    #current batch_size
    b_size = None
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 100
    early_stopping = 10
    random_seed = 11


    #inference='crf'
    inference='reinforced-decoder-rnn'
    gamma = 0.7
    n_step = 5

    decoding = 'greedy'
    #decoding = 'beamsearch'
    #beamsize = 3

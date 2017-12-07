class Configuration(object):
    """Model hyper parameters and data information"""

    """ Path to different files """
    source_alphabet = './data/EnPe/source.txt'
    target_alphabet = './data/EnPe/target.txt'
    train_set = './data/EnPe/mytrain.txt'
    dev_set = './data/EnPe/mydev.txt'
    test_set = './data/EnPe/mytest.txt'
    end_symbol = '#'

    """ Neural hyper params """
    s_embedding_size = 100
    s_alphabet_size = None
    t_embedding_size = 100
    t_alphabet_size = None
    max_length = 32
    h_units = 200
    batch_size = 32
    #current batch_size
    b_size = None
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 100
    early_stopping = 4
    random_seed = 1001


    #inference='crf'
    inference='reinforced-decoder-rnn'
    gamma = 0.45
    n_step = 3

    #decoding = 'greedy'
    #decoding = 'beamsearch'
    #beamsize = 2

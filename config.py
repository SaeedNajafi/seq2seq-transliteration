class Configuration(object):
    """Model hyper parameters and data information"""

    #Path to different files
    source_alphabet = './data/source.txt'
    target_alphabet = './data/target.txt'
    train_set = './data/train.txt'
    dev_set = './data/dev.txt'
    test_set = './data/test.txt'

    #Neural hyper params
    s_embedding_size = 16
    t_embedding_size = 16
    max_length = 64
    h_units = 32
    batch_size = 32
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 50
    early_stopping = 2

    #For decoding
    beamsize = 4

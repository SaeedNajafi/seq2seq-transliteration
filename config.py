class Configuration(object):
    """Model hyper parameters and data information"""

    #Path to different files
    source_alphabet = './data/EnPe/source.txt'
    target_alphabet = './data/EnPe/target.txt'
    train_set = './data/EnPe/train-pe.txt'
    dev_set = './data/EnPe/dev-pe.txt'
    test_set = './data/EnPe/enpe11.txt'

    #Neural hyper params
    s_embedding_size = 16
    s_alphabet_size = None
    t_embedding_size = 16
    t_alphabet_size = None

    max_length = 32
    h_units = 32

    batch_size = 32
    #current batch_size
    b_size = None

    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 50
    early_stopping = 2

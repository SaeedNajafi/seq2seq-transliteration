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
    t_embedding_size = 100
    max_length = 32
    h_units = 200
    batch_size = 128

    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 100
    early_stopping = 5
    runs=10
    gamma = 0.7
    n_step = 3

    #inference = "CRF"
    #inference = "RNN"
    #inference = "AC-RNN"
    #inference = "DIF-SCH"
    #inference = "SCH"
    #inference = "R-RNN"
    #inference = "BR-RNN"

    beamsize = 4

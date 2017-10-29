from config import Configuration
from model import Model
import utils as ut
import numpy as np
import tensorflow as tf
import sys
import time
import os

def run_epoch(config, model, session, X, X_length, X_mask, Y, Y_length, Y_mask):

    # We're interested in keeping track of the loss during training
    total_loss = []
    total_steps = int(np.ceil(len(X) / float(config.batch_size)))
    data = ut.data_iterator(X, X_length, X_mask, Y, Y_length, Y_mask, config.batch_size, True)
    for step, (X_in, X_length_in, X_mask_in, Y_in, Y_length_in, Y_mask_in) in enumerate(data):
        feed = model.create_feed_dict(
                    X=X_in,
                    X_length=X_length_in,
                    X_mask=X_mask_in,
                    Y=Y_in,
                    Y_length=Y_length_in,
                    Y_mask=Y_mask_in,
                    dropout=config.dropout
                    )
        loss , _ = session.run([model.loss, model.train_op], feed_dict=feed)
        total_loss.append(loss)
        ##
        sys.stdout.write('\r{} / {} : loss = {}'.format(step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()

    return np.mean(total_loss)

def predict(config, model, session, X, X_length, X_mask):
    results = []
    lengths = []
    total_steps = int(np.ceil(len(X) / float(config.batch_size)))
    data = ut.data_iterator(X, X_length, X_mask, Y, Y_length, Y_mask, config.batch_size, True)
    for step, (X_in, X_length_in, X_mask_in, Y_in, Y_length_in, Y_mask_in) in enumerate(data):
        feed = model.create_feed_dict(
                    X=X_in,
                    X_length=X_length_in,
                    X_mask=X_mask_in,
                    Y=None,
                    Y_length=None,
                    Y_mask=None,
                    dropout=1
                    )

        batch_predicted_indices, batch_lengths = session.run([model.outputs, model.lengths], feed_dict=feed)
        results.append(batch_predicted_indices[0])
        lengths.append(batch_lengths[0])

    return results, lengths

def save_predictions(
            config,
            predictions,
            predictions_length,
            filename,
            X
            X_length,
            Y
            Y_length,
            s_num_to_char,
            t_num_to_char
            ):

    """Saves predictions to the provided file."""
    with open(filename, "wb") as f:
        for batch_index in range(len(predictions)):
            batch_predictions = predictions[batch_index]
            b_size = len(batch_predictions)
            for word_index in range(b_size):
                for char_index in range(config.max_length):
                    ad = (batch_index * config.batch_size) + word_index
                    s_to_file = ""
                    t_to_file = ""
                    p_to_file = ""
                    if(char_index < X_length[ad]):
                        s_to_file += str(s_num_to_char[X[ad][char_index]])

                    if(char_index < predictions_length[ad]):
                        p_to_file += str(t_num_to_char[predictions[ad][char_index]])

                    if Y is not None:
                        if(char_index < Y_length[ad]):
                            t_to_file += str(t_num_to_char[Y[ad][char_index]])

                to_file = s_to_file + "\t" + t_to_file + "\t" + p_to_file+ "\n"
                f.write(to_file)
    return

#Recursive Levenshtein Function in Python
def LD(y1, y2):
    if y1 == "":
        return len(y2)

    if y2 == "":
        return len(y1)

    if y1[-1] == y2[-1]:
        cost = 0
    else:
        cost = 1

    dist = min([LD(y1[:-1], y2)+ 1,
               LD(y1, y2[:-1])+ 1,
               LD(y1[:-1], y2[:-1]) + cost])
    return dist

def evaluate():
    with open(filename, "r") as f:
        lines = f.readlines()
        dist = 0.0
        for each in lines:
            each = each.strip()
            line = each.split("\t")
            if len(line)==2:
                dist += LD(line[1], line[2])
            else:
                print "Wrong input file format for evaluating!"
                exit()
        cost = float(dist/len(lines))
    return cost

def run(mode):
    """run the model's implementation.
    """
    config = Configuration()
    data = ut.load_data(config, mode)
    model = Model(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)

        if mode=='train':
            best_dev_cost = float('inf')
            best_dev_epoch = 0
            first_start = time.time()
            for epoch in xrange(config.max_epochs):
                print
                print 'Epoch {}'.format(epoch)

                start = time.time()
                train_loss = run_epoch(
                                    config,
                                    model,
                                    session,
                                    data['train_data']['X'],
                                    data['train_data']['X_length'],
                                    data['train_data']['X_mask'],
                                    data['train_data']['Y'],
                                    data['train_data']['Y_length'],
                                    data['train_data']['Y_mask']
                                    )

                predictions, predictions_length = predict(
                                                    config,
                                                    model,
                                                    session,
                                                    data['dev_data']['X'],
                                                    data['dev_data']['X_length'],
                                                    data['dev_data']['X_mask']
                                                    )

                print 'Training loss: {}'.format(train_loss)
                save_predictions(
                                config,
                                predictions,
                                predictions_length,
                                "temp.predicted",
                                data['dev_data']['X'],
                                data['dev_data']['X_length'],
                                data['dev_data']['Y'],
                                data['dev_data']['Y_length'],
                                data['s_num_to_char'],
                                data['t_num_to_char']
                                )

                dev_cost = evaluate()
                print 'Validation cost: {}'.format(dev_cost)

                if  dev_cost < best_dev_cost:
                    best_dev_cost = dev_cost
                    best_dev_epoch = epoch
                    if not os.path.exists("./weights"):
                    	os.makedirs("./weights")
                    saver.save(session, './weights/weights')

                # For early stopping which is kind of regularization for network.
                if epoch - best_dev_epoch > config.early_stopping:
                    break
                    ###

                print 'Epoch training time: {} seconds'.format(time.time() - start)

            print 'Total training time: {} seconds'.format(time.time() - first_start)

        elif mode=='test':
            saver.restore(session, './weights/weights')
            print
            print
            print 'Test'
            start = time.time()
            predictions, predictions_length = predict(
                                                    config,
                                                    model,
                                                    session,
                                                    data['test_data']['X'],
                                                    data['test_data']['X_length'],
                                                    data['test_data']['X_mask']
                                                    )

            print 'Total prediction time: {} seconds'.format(time.time() - start)
            print 'Writing predictions to test.predicted'
            save_predictions(
                            config,
                            predictions,
                            predictions_length,
                            "test.predicted",
                            data['test_data']['X'],
                            data['test_data']['X_length'],
                            None,
                            None,
                            data['s_num_to_char'],
                            data['t_num_to_char']
                            )
        else:
            print "Specify an option: train or test?"
            exit()

if __name__ == "__main__":
    #sys.argv[1] is 'test' or 'train'
    run(sys.argv[1])

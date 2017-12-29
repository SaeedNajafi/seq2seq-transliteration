from config import Configuration
from model import Model
import utils as ut
import numpy as np
import tensorflow as tf
import sys
import time
import os

def train(config, model, session, X, X_length, X_mask, alpha, Y, Y_length, Y_mask):

    #We're interested in keeping track of the loss during training
    total_loss = []
    V_total_loss = []
    total_steps = int(np.ceil(len(X) / float(config.batch_size)))
    data = ut.data_iterator(X, X_length, X_mask, Y, Y_length, Y_mask, config.batch_size, True)
    for step, (X_in, X_length_in, X_mask_in, Y_in, Y_length_in, Y_mask_in) in enumerate(data):
        feed = model.create_feed_dict(
                    X=X_in,
                    X_length=X_length_in,
                    X_mask=X_mask_in,
                    dropout=config.dropout,
                    alpha=alpha,
                    Y=Y_in,
                    Y_length=Y_length_in,
                    Y_mask=Y_mask_in
                    )

        if alpha==0.0:
            loss , _ = session.run([model.loss, model.train_op], feed_dict=feed)
            total_loss.append(loss)

            ##
            sys.stdout.write('\r{} / {} : loss = {}'.format(step, total_steps, np.mean(total_loss)))
            sys.stdout.flush()

        else:
            V_loss, loss, _, _ = session.run([model.V_loss, model.loss, model.V_train_op, model.train_op], feed_dict=feed)
            total_loss.append(loss)
            V_total_loss.append(V_loss)

            ##
            sys.stdout.write('\r{} / {} : loss = {} | V loss = {}'.format(
                                                                            step,
                                                                            total_steps,
                                                                            np.mean(total_loss),
                                                                            np.mean(V_total_loss)
                                                                        )
                             )
            sys.stdout.flush()

    if alpha==0.0:
        return np.mean(total_loss), 0
    else:
        return np.mean(total_loss), np.mean(V_total_loss)

def predict(config, model, session, X, X_length, X_mask, Y=None, Y_length=None, Y_mask=None):
    results = []
    total_steps = int(np.ceil(len(X) / float(config.batch_size)))
    data = ut.data_iterator(X, X_length, X_mask, Y, Y_length, Y_mask, config.batch_size, False)
    for step, (X_in, X_length_in, X_mask_in, Y_in, Y_length_in, Y_mask_in) in enumerate(data):
        feed = model.create_feed_dict(
                    X=X_in,
                    X_length=X_length_in,
                    X_mask=X_mask_in,
                    dropout=1,
                    alpha=0.0
                    )

        if config.inference=="RNN" or config.inference=="AC-RNN":
            batch_predicted_indices = session.run([model.outputs], feed_dict=feed)
            results.append(batch_predicted_indices[0])

        elif config.inference=="CRF":
            unary_scores, transition_params = session.run([model.M, model.crf_transition_params], feed_dict=feed)
            batch_results = []
            for unary_scores_each in unary_scores:
                # Compute the highest score and its tag sequence.
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(unary_scores_each, transition_params)
                predicted_indices = viterbi_sequence
                batch_results.append(predicted_indices)

            results.append(batch_results)
    return results

def save_predictions(
            config,
            predictions,
            filename,
            X,
            X_length,
            s_num_to_char,
            t_num_to_char,
            Y=None,
            Y_length=None
            ):

    """Saves predictions to the provided file."""
    with open(filename, "wb") as f:

        for batch_index in range(len(predictions)):
            batch_predictions = predictions[batch_index]
            b_size = len(batch_predictions)
            for word_index in range(b_size):
                ad = (batch_index * config.batch_size) + word_index
                s_to_file = ""
                t_to_file = ""
                p_to_file = ""
                p_end = False
                for char_index in range(config.max_length):

                    #not considering the end sign
                    if(char_index < X_length[ad] - 1):
                        s_to_file += str(s_num_to_char[X[ad][char_index]].encode('utf8'))

                    if(str(t_num_to_char[batch_predictions[word_index][char_index]].encode('utf8'))!=config.end_symbol and p_end==False):
                        p_to_file += str(t_num_to_char[batch_predictions[word_index][char_index]].encode('utf8'))
                    else: p_end = True

                    if Y is not None:
                        if(char_index < Y_length[ad] - 1):
                            t_to_file += str(t_num_to_char[Y[ad][char_index]].encode('utf8'))

                if Y is not None:
                    to_file = s_to_file + "\t" + t_to_file + "\t" + p_to_file+ "\n"
                else:
                    to_file = p_to_file+ "\n"


                f.write(to_file)
    return

#Recursive Levenshtein Distance Function in Python
def LD(str1, str2):
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in xrange(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in xrange(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in xrange(1, len(str2) + 1):
        for y in xrange(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])

def avg_edit_distance(fileName):
    with open(fileName, "r") as f:
        lines = f.readlines()
        dist = 0.0
        for each in lines:
            each = each.strip()
            line = each.split("\t")
            if len(line)==3:
                ref = list(line[1].decode('utf8'))
                pred = list(line[2].decode('utf8'))
                dist += LD(ref, pred)
            else:
                print "Wrong input file format for evaluating!"
                exit()

        cost = float(dist/len(lines))
    return cost

def accuracy(fileName):
    pred_lines = open(fileName, 'r').readlines()
    total_source = 0.0
    total_correct = 0.0
    for each in pred_lines:
        each = each.strip()
        line = each.split("\t")
        if len(line)==3:
            ref = list(line[1].decode('utf8'))
            pred = list(line[2].decode('utf8'))
            total_source += 1
            if ref == pred:
                total_correct += 1
        else:
            print "Wrong input file format for evaluating!"
            exit()

    return (total_correct/total_source)*100


def run_model():
    config = Configuration()
    data = ut.load(config, 'train')
    path = "./EnPe_results"
    if not os.path.exists(path):
        os.makedirs(path)

    #alpha shows how much we care about the reinforcement learning.
    alpha = 0.0
    save_epoch = -1
    model_name = config.inference
    for i in range(config.runs):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            run = i + 1
            seed = run**3
            tf.set_random_seed(seed)
            np.random.seed(seed)
            model = Model(config, seed)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                best_dev_cost = float('inf')
                best_dev_epoch = 0
                session.run(init)

                if model_name=='AC-RNN':
                    save_epoch = -1
                    saver.restore(session, path + '/' + 'RNN' + '.' + str(run) + '/weights')
                    optimizer_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adam_optimizer")
                    session.run(tf.variables_initializer(optimizer_scope))

                first_start = time.time()
                epoch = 0
                while (epoch<config.max_epochs):
                    if model_name=='AC-RNN':
                        alpha = np.minimum(0.95, 0.5 + epoch * 0.05)

                    print
                    print 'Model:{} Run:{} Epoch:{}'.format(model_name, run, epoch)
                    start = time.time()
                    train_loss , V_train_loss = train(
                                                    config,
                                                    model,
                                                    session,
                                                    data['train_data']['X'],
                                                    data['train_data']['X_length'],
                                                    data['train_data']['X_mask'],
                                                    alpha,
                                                    data['train_data']['Y'],
                                                    data['train_data']['Y_length'],
                                                    data['train_data']['Y_mask']
                                                    )
                    if epoch > save_epoch:
                        predictions = predict(
                                             config,
                                             model,
                                             session,
                                             data['dev_data']['X'],
                                             data['dev_data']['X_length'],
                                             data['dev_data']['X_mask'],
                                             data['dev_data']['Y'],
                                             data['dev_data']['Y_length'],
                                             data['dev_data']['Y_mask']
                                             )

                        print '\rTraining loss: {}'.format(train_loss)
                        if alpha!=0.0: print 'V Training loss: {}'.format(V_train_loss)
                        save_predictions(
                                        config,
                                        predictions,
                                        "temp.predicted",
                                        data['dev_data']['X'],
                                        data['dev_data']['X_length'],
                                        data['s_id_to_char'],
                                        data['t_id_to_char'],
                                        data['dev_data']['Y'],
                                        data['dev_data']['Y_length']
                                        )

                        dev_cost = 100 - accuracy("temp.predicted")
                        print 'Validation cost: {}'.format(dev_cost)
                        if  dev_cost < best_dev_cost:
                            best_dev_cost = dev_cost
                            best_dev_epoch = epoch
                            if not os.path.exists(path + '/' + model_name + '.' + str(run)):
                                os.makedirs(path + '/' + model_name + '.' + str(run))
                            saver.save(session, path + '/' + model_name + '.' + str(run) + '/weights')

                        #For early stopping which is kind of regularization for network.
                        if epoch - best_dev_epoch > config.early_stopping:
                            break
                            ###

                        print 'Epoch training time: {} seconds'.format(time.time() - start)
                    epoch += 1
                print 'Total training time: {} seconds'.format(time.time() - first_start)

                saver.restore(session, path + '/' + model_name + '.' + str(run) + '/weights')
                print
                print 'Model:{} Run:{} Dev'.format(model_name, run)
                start = time.time()
                predictions = predict(
                                     config,
                                     model,
                                     session,
                                     data['dev_data']['X'],
                                     data['dev_data']['X_length'],
                                     data['dev_data']['X_mask'],
                                     data['dev_data']['Y'],
                                     data['dev_data']['Y_length'],
                                     data['dev_data']['Y_mask']
                                     )

                print 'Total prediction time: {} seconds'.format(time.time() - start)
                print 'Writing predictions to dev.predicted'
                save_predictions(
                                config,
                                predictions,
                                path + '/' + model_name + '.' + str(run) + '.' + "dev.predicted",
                                data['dev_data']['X'],
                                data['dev_data']['X_length'],
                                data['s_id_to_char'],
                                data['t_id_to_char'],
                                data['dev_data']['Y'],
                                data['dev_data']['Y_length']
                                )

		print
                data = ut.load(config, 'test')
                print 'Model:{} Run:{} Test'.format(model_name, run)
                predictions = predict(
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
                                path + '/' + model_name + '.' + str(run) + '.' + "test.predicted",
                                data['test_data']['X'],
                                data['test_data']['X_length'],
                                data['s_id_to_char'],
                                data['t_id_to_char']
                                )
    return

if __name__ == "__main__":
    run_model()

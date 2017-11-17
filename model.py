import tensorflow as tf
import numpy as np

class Model(object):
    """ Implements the sequence to sequence model for transliteration. """

    def __init__(self, config):
        """Constructs the network using the helper functions defined below."""

        self.placeholders(config)

        s_embed, t_embed = self.embeddings(config)

        H = self.encoder(s_embed, config)

        M = self.decoder(H, t_embed, config)

        self.outputs = self.greedy_decoding(H, config)

        self.loss = tf.contrib.seq2seq.sequence_loss(
                                    logits=M,
                                    targets=self.Y_placeholder,
                                    weights=self.Y_mask_placeholder,
                                    average_across_timesteps=True,
                                    average_across_batch=True
                                    )
        self.train_op = self.add_training_op(self.loss, config)

        return

    def placeholders(self, config):
        """Generates placeholder variables to represent the input tensors
        These placeholders are used as inputs by the rest of the model building code
        and will be fed data during training.
        """

        self.X_placeholder = tf.placeholder(tf.int32, shape=(None, config.max_length))
        self.X_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.X_mask_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, config.max_length))
        self.Y_placeholder = tf.placeholder(tf.int32, shape=(None, config.max_length))
        self.Y_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.Y_mask_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, config.max_length))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())
        return

    def create_feed_dict(self, X, X_length, X_mask, dropout, Y=None, Y_length=None, Y_mask=None):
        """Creates the feed_dict.
        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }
        """

        feed_dict={
            self.X_placeholder: X,
            self.X_length_placeholder: X_length,
            self.X_mask_placeholder: X_mask,
            self.dropout_placeholder: dropout
            }

        if Y is not None:
            feed_dict[self.Y_placeholder] = Y
            feed_dict[self.Y_length_placeholder] = Y_length
            feed_dict[self.Y_mask_placeholder] = Y_mask

        return feed_dict

    def embeddings(self, config):
        """Add embedding layer that maps from characters to vectors.
        """

        boundry = np.sqrt(np.divide(3.0, config.s_embedding_size))
        s_char_vectors = np.random.uniform(
                                low=-boundry, high=boundry,
                                size=(config.s_alphabet_size, config.s_embedding_size)
                                )

        with tf.variable_scope("s_embeddings"):
            s_lookup_table = tf.Variable(
                               	    s_char_vectors,
                                    name="s_lookup_table",
                                    dtype=tf.float32
                                    )

            s_embeddings = tf.nn.embedding_lookup(
                                s_lookup_table,
                                self.X_placeholder
                                )

        boundry = np.sqrt(np.divide(3.0, config.t_embedding_size))
        t_char_vectors = np.random.uniform(
                                low=-boundry, high=boundry,
                                size=(config.t_alphabet_size, config.t_embedding_size)
                                )

        with tf.variable_scope("t_embeddings"):
            self.t_lookup_table = tf.Variable(
                               	    t_char_vectors,
                                    name="t_lookup_table",
                                    dtype=tf.float32
                                    )

            t_embeddings = tf.nn.embedding_lookup(
                                self.t_lookup_table,
                                self.Y_placeholder
                                )

        config.b_size = tf.shape(s_embeddings)[0]
        return s_embeddings, t_embeddings

    def xavier_initializer (self, shape, **kargs):
        """Defines an initializer for the Xavier distribution.
        This function will be used as a variable initializer.
        Args:
          shape: Tuple or 1-d array that specifies dimensions of the requested tensor.
        Returns:
          out: tf.Tensor of specified shape sampled from Xavier distribution.
        """

        sum_of_dimensions = tf.reduce_sum(shape)
        epsilon = tf.cast(
                        tf.sqrt( tf.divide(6, sum_of_dimensions) ),
                        tf.float32
                        )

        out = tf.random_uniform(shape,
                                minval=-epsilon,
                                maxval=epsilon,
                                dtype=tf.float32
                                )

        return out

    def encoder(self, s_embed, config):

        with tf.variable_scope('encoder_rnn') as scope:
            forward_encoder_lstm = tf.contrib.rnn.LSTMCell(
                                            num_units=config.h_units,
                                            use_peepholes=False,
                                            cell_clip=None,
                                            initializer=self.xavier_initializer,
                                            num_proj=None,
                                            proj_clip=None,
                                            num_unit_shards=None,
                                            num_proj_shards=None,
                                            forget_bias=1.0,
                                            state_is_tuple=True,
                                            activation=tf.tanh
                                            )

            backward_encoder_lstm = tf.contrib.rnn.LSTMCell(
                                            num_units=config.h_units,
                                            use_peepholes=False,
                                            cell_clip=None,
                                            initializer=self.xavier_initializer,
                                            num_proj=None,
                                            proj_clip=None,
                                            num_unit_shards=None,
                                            num_proj_shards=None,
                                           	forget_bias=1.0,
                                            state_is_tuple=True,
                                            activation=tf.tanh
                                            )

            (h_fw, h_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                        forward_encoder_lstm,
                                        backward_encoder_lstm,
                                        s_embed,
                                        sequence_length=self.X_length_placeholder,
                                        initial_state_fw=None,
                                        initial_state_bw=None,
                                        dtype=tf.float32,
                                        parallel_iterations=None,
                                        swap_memory=False,
                                        time_major=False,
                                        scope=scope
                                        )

        encoder_final_hs = tf.concat([h_fw, h_bw], axis=2)

        #apply dropout
        dropped_encoder_final_hs = tf.nn.dropout(
                                        encoder_final_hs,
                                        self.dropout_placeholder
                                    )

        """hidden layer"""
        with tf.variable_scope("encoder_hidden"):
            W_hidden = tf.get_variable(
                            "W_hidden",
                            (2 * config.h_units, config.h_units),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_hidden = tf.get_variable(
                            "b_hidden",
                            (config.h_units,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )


            H = tf.add(
                        tf.matmul(
                            tf.reshape(
                                dropped_encoder_final_hs,
                                (-1, 2  *  config.h_units)
                            ),
                            W_hidden
                        ),
                        b_hidden
                    )
            H = tf.tanh(H)
            H = tf.reshape(H, (-1, config.max_length, config.h_units))

        return H

    def decoder(self, H, t_embed, config):

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                            "W_softmax",
                            (2 * config.h_units, config.t_alphabet_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (config.t_alphabet_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

        with tf.variable_scope('decoder_rnn') as scope:
            self.decoder_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=config.h_units,
                                        use_peepholes=False,
                                        cell_clip=None,
                                        initializer=self.xavier_initializer,
                                        num_proj=None,
                                        proj_clip=None,
                                        num_unit_shards=None,
                                        num_proj_shards=None,
                                        forget_bias=1.0,
                                        state_is_tuple=True,
                                        activation=tf.tanh
                                        )

        GO_symbol = tf.zeros((config.b_size, config.t_embedding_size), dtype=tf.float32)
        t_embed_tra = tf.transpose(t_embed, [1,0,2])
        H_tra = tf.transpose(H, [1,0,2])

        M = []
        with tf.variable_scope('decoder_rnn') as scope:
            initial_state = self.decoder_lstm.zero_state(config.b_size, tf.float32)
            for time_index in range(config.max_length):
                if time_index==0:
                    output, state = self.decoder_lstm(GO_symbol, initial_state)
                else:
                    scope.reuse_variables()
                    output, state = self.decoder_lstm(t_embed_tra[time_index-1], state)

                output_dropped = tf.nn.dropout(output, self.dropout_placeholder)
                H_and_output = tf.concat([H_tra[time_index], output_dropped], axis=1)
                m = tf.add(tf.matmul(H_and_output, W_softmax), b_softmax)
                M.append(m)

            M = tf.stack(M, axis=1)

        return M

    def greedy_decoding(self, H, config):

        """Reload softmax prediction layer"""
        with tf.variable_scope("softmax", reuse=True):
            W_softmax = tf.get_variable("W_softmax")
            b_softmax = tf.get_variable("b_softmax")

        GO_symbol = tf.zeros((config.b_size, config.t_embedding_size), dtype=tf.float32)
        initial_state = self.decoder_lstm.zero_state(config.b_size, tf.float32)
        H_tra = tf.transpose(H, [1,0,2])

        outputs = []
        with tf.variable_scope("decoder_rnn", reuse=True) as scope:
            for time_index in range(config.max_length):
                if time_index==0:
                    output, state = self.decoder_lstm(GO_symbol, initial_state)
                else:
                    output, state = self.decoder_lstm(prev_output, state)

                output_dropped = tf.nn.dropout(output, self.dropout_placeholder)
                H_and_output = tf.concat([H_tra[time_index], output_dropped], axis=1)
                m = tf.add(tf.matmul(H_and_output, W_softmax), b_softmax)
                probs = tf.nn.softmax(m)
                predicted_indices = tf.argmax(probs, axis=1)
                outputs.append(predicted_indices)
                prev_output = tf.nn.embedding_lookup(self.t_lookup_table, predicted_indices)

            outputs = tf.stack(outputs, axis=1)

        return outputs

    def add_training_op(self, loss, config):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """

        #we use adam optimizer
        with tf.variable_scope("adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, config.max_gradient_norm)
        train_operation = optimizer.apply_gradients(zip(clipped_gradients, variables))

        return train_operation

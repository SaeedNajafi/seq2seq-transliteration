import tensorflow as tf
import numpy as np

class Model(object):
    """ Implements the sequence to sequence model for transliteration. """

    def __init__(self, config):
        """Constructs the network using the helper functions defined below."""

        self.placeholders(config)

        s_embed, t_embed = self.embeddings(config)

        C = self.encoder(s_embed, config)

        if config.inference=='reinforced-decoder-rnn':

            self.reinforced_decoder(C, t_embed, config)

            with tf.variable_scope("baseline_adam_optimizer"):
                self.baseline_train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.baseline_loss)

            if config.decoding=='greedy':
                self.outputs = self.greedy_decoding(C, config)

            elif config.decoding=='beamsearch':
                self.outputs = self.beam_decoding(C, config)

        elif config.inference=='crf':
            self.crf_decoder(C, config)
            self.loss = tf.reduce_mean(-self.crf_log_likelihood)

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
        self.pretrain_placeholder = tf.placeholder(dtype=tf.bool, shape=())
        return

    def create_feed_dict(self, X, X_length, X_mask, dropout, pretrain, Y=None, Y_length=None, Y_mask=None):
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
            self.dropout_placeholder: dropout,
            self.pretrain_placeholder: pretrain
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

            #local soft attention | window-based local attention
            C = []
            H_tra = tf.transpose(H, [1,0,2])
            for time_index in range(config.max_length):
                prev_c = H_tra[time_index] - H_tra[time_index]
                next_c = H_tra[time_index] - H_tra[time_index]
                curr_c = H_tra[time_index]
                if time_index==0:
                    next_c = H_tra[time_index + 1]
                elif time_index==config.max_length-1:
                    prev_c = H_tra[time_index - 1]
                else:
                    prev_c = H_tra[time_index - 1]
                    next_c = H_tra[time_index + 1]

                c = tf.concat([prev_c, curr_c, next_c], axis=1)
                C.append(c)

            C = tf.stack(C, axis=1)
        return C

    def crf_decoder(self, C, config):

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                                "W_softmax",
                                (3 * config.h_units, config.t_alphabet_size),
                                tf.float32,
                                self.xavier_initializer
                                )

            b_softmax = tf.get_variable(
                                "b_softmax",
                                (config.t_alphabet_size,),
                                tf.float32,
                                tf.constant_initializer(0.0)
                                )


        M = tf.add(tf.matmul(tf.reshape(C, (-1, 3 * config.h_units)), W_softmax), b_softmax)

        self.M = tf.reshape(M, (-1, config.max_length, config.t_alphabet_size))
        self.crf_log_likelihood, self.crf_transition_params = tf.contrib.crf.crf_log_likelihood(
                                                                    self.M,
                                                                    self.Y_placeholder,
                                                                    self.Y_length_placeholder - self.Y_length_placeholder + config.max_length
                                                                    )
        return

    def reinforced_decoder(self, C, t_embed, config):

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                            "W_softmax",
                            (4 * config.h_units, config.t_alphabet_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (config.t_alphabet_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

        with tf.variable_scope("baseline"):
            W1_baseline = tf.get_variable(
                            "W1_baseline",
                            (4 * config.h_units, 32),
                            tf.float32,
                            self.xavier_initializer
                            )

            b1_baseline = tf.get_variable(
                            "b1_baseline",
                            (32,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

            W2_baseline = tf.get_variable(
                            "W2_baseline",
                            (32, 1),
                            tf.float32,
                            self.xavier_initializer
                            )

            b2_baseline = tf.get_variable(
                            "b2_baseline",
                            (1,),
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
        C_tra = tf.transpose(C, [1,0,2])

        def maximum_likelihood():
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


                    C_and_output = tf.concat([C_tra[time_index], output_dropped], axis=1)
                    m = tf.add(tf.matmul(C_and_output, W_softmax), b_softmax)

                    M.append(m)

            M = tf.stack(M, axis=1)
            cross_entropy_loss = tf.contrib.seq2seq.sequence_loss(
                                                        logits=M,
                                                        targets=self.Y_placeholder,
                                                        weights=self.Y_mask_placeholder,
                                                        average_across_timesteps=True,
                                                        average_across_batch=True
                                                        )

            #b2_baseline is just a dummy loss added for coding purpose.
            return cross_entropy_loss, b2_baseline

        def actor_critic():
            Policies = []
            Baselines = []
            with tf.variable_scope('decoder_rnn') as scope:
                initial_state = self.decoder_lstm.zero_state(config.b_size, tf.float32)
                for time_index in range(config.max_length):
                    if time_index==0:
                        output, state = self.decoder_lstm(GO_symbol, initial_state)
                    else:
                        scope.reuse_variables()
                        output, state = self.decoder_lstm(prev_output, state)

                    output_dropped = tf.nn.dropout(output, self.dropout_placeholder)

                    C_and_output = tf.concat([C_tra[time_index], output_dropped], axis=1)


                    #forward pass for the baseline estimation
                    baseline = tf.add(tf.matmul(tf.stop_gradient(C_and_output), W1_baseline), b1_baseline)
                    baseline = tf.add(tf.matmul(baseline, W2_baseline), b2_baseline)
                    Baselines.append(baseline)

                    m = tf.add(tf.matmul(C_and_output, W_softmax), b_softmax)
                    policy = tf.nn.softmax(m)
                    prev_output = tf.matmul(tf.nn.softmax(10000.0 * m), self.t_lookup_table)
                    Policies.append(policy)

            Policies = tf.stack(Policies, axis=1)
            Baselines = tf.stack(Baselines, axis=1)

            Baselines = tf.reshape(Baselines, (-1, config.max_length))

            is_true_tag = tf.cast(tf.equal(tf.cast(self.Y_placeholder, tf.int64), tf.argmax(Policies, axis=2)), tf.float32)
            Rewards = 2 * is_true_tag - 1.0
            Rewards = tf.multiply(Rewards, self.Y_mask_placeholder)
            Baselines = tf.multiply(Baselines, self.Y_mask_placeholder)

            Rewards_t = tf.transpose(Rewards, [1,0])
            Baselines_t = tf.transpose(Baselines, [1,0])
            Returns = []

            zeros = tf.cast(self.Y_length_placeholder - self.Y_length_placeholder, tf.float32)
            for t in range(config.max_length):
                ret = zeros
                if t < config.max_length - config.n_step:
                    for i in range(config.n_step):
                        ret += (config.gamma ** i) * Rewards_t[t + i]
                        if i == config.n_step - 1:
                            ret += (config.gamma ** config.n_step) * Baselines_t[t + config.n_step]

                Returns.append(ret)

            Returns = tf.stack(Returns, axis=1)

            Objective = tf.log(tf.reduce_max(Policies, axis=2)) * tf.stop_gradient(Returns - Baselines)
            Objective_masked = tf.multiply(Objective, self.Y_mask_placeholder)

            baseline_loss = tf.reduce_mean(tf.pow(tf.stop_gradient(Returns) - Baselines, 2) * self.Y_mask_placeholder) / 2.0

            actor_loss = -tf.reduce_mean(Objective_masked)
            return actor_loss, baseline_loss

        self.loss, self.baseline_loss = tf.cond(self.pretrain_placeholder, maximum_likelihood, actor_critic)

        return

    def greedy_decoding(self, C, config):

        """Reload softmax prediction layer"""
        with tf.variable_scope("softmax", reuse=True):
            W_softmax = tf.get_variable("W_softmax")
            b_softmax = tf.get_variable("b_softmax")

        GO_symbol = tf.zeros((config.b_size, config.t_embedding_size), dtype=tf.float32)
        initial_state = self.decoder_lstm.zero_state(config.b_size, tf.float32)
        C_tra = tf.transpose(C, [1,0,2])

        outputs = []
        with tf.variable_scope("decoder_rnn", reuse=True) as scope:
            for time_index in range(config.max_length):
                if time_index==0:
                    output, state = self.decoder_lstm(GO_symbol, initial_state)
                else:
                    output, state = self.decoder_lstm(prev_output, state)

                C_and_output = tf.concat([C_tra[time_index], output], axis=1)
                m = tf.add(tf.matmul(C_and_output, W_softmax), b_softmax)


                probs = tf.nn.softmax(m)
                predicted_indices = tf.argmax(probs, axis=1)
                outputs.append(predicted_indices)
                prev_output = tf.nn.embedding_lookup(self.t_lookup_table, predicted_indices)

            outputs = tf.stack(outputs, axis=1)

        return outputs

    def beam_decoding(self, C, config):

        """Reload softmax prediction layer"""
        with tf.variable_scope("softmax", reuse=True):
            W_softmax = tf.get_variable("W_softmax")
            b_softmax = tf.get_variable("b_softmax")

        GO_symbol = tf.zeros((config.b_size, config.t_embedding_size), dtype=tf.float32)
        initial_state = self.decoder_lstm.zero_state(config.b_size, tf.float32)
        C_tra = tf.transpose(C, [1,0,2])

        """ we will need index to select top ranked beamsize stuff"""
        #batch index
        b_index = tf.reshape(tf.range(0, config.b_size),(config.b_size, 1))

        #beam index
        be_index = tf.constant(
                                config.beamsize * config.beamsize,
                                dtype=tf.int32,
                                shape=(1, config.beamsize)
                                )


        with tf.variable_scope("decoder_rnn", reuse=True) as scope:
            for time_index in range(config.max_length):
                if time_index==0:
                    output, (c_state, m_state) = self.decoder_lstm(GO_symbol, initial_state)

                    C_and_output = tf.concat([C_tra[time_index], output], axis=1)
                    pred = tf.add(tf.matmul(C_and_output, W_softmax), b_softmax)

                    predictions = tf.nn.softmax(pred)
                    probs, indices = tf.nn.top_k(predictions, k=config.beamsize, sorted=True)
                    prev_indices = indices
                    beam = tf.expand_dims(indices, axis=2)
                    prev_probs = tf.log(probs)
                    prev_c_states = [c_state for i in range(config.beamsize)]
                    prev_c_states = tf.stack(prev_c_states, axis=1)
                    prev_m_states = [m_state for i in range(config.beamsize)]
                    prev_m_states = tf.stack(prev_m_states, axis=1)

                else:
                    prev_indices_t = tf.transpose(prev_indices, [1,0])
                    prev_probs_t = tf.transpose(prev_probs, [1,0])
                    prev_c_states_t = tf.transpose(prev_c_states, [1,0,2])
                    prev_m_states_t = tf.transpose(prev_m_states, [1,0,2])
                    beam_t = tf.transpose(beam, [1,0,2])

                    probs_candidates = []
                    indices_candidates = []
                    beam_candidates = []
                    c_state_candidates = []
                    m_state_candidates = []
                    for b in range(config.beamsize):
                        prev_output = tf.nn.embedding_lookup(self.t_lookup_table, prev_indices_t[b])
                        output, (c_state, m_state) = self.decoder_lstm(
                                                        prev_output,
                                                        (prev_c_states_t[b],prev_m_states_t[b])
                                                        )


                        C_and_output = tf.concat([C_tra[time_index], output], axis=1)
                        pred = tf.add(tf.matmul(C_and_output, W_softmax), b_softmax)


                        predictions = tf.nn.softmax(pred)
                        probs, indices = tf.nn.top_k(predictions, k=config.beamsize, sorted=True)
                        probs_t = tf.transpose(probs, [1,0])
                        indices_t = tf.transpose(indices, [1,0])
                        for bb in range(config.beamsize):
                            probs_candidates.append(tf.add(prev_probs_t[b], tf.log(probs_t[bb])))
                            indices_candidates.append(indices_t[bb])
                            beam_candidates.append(tf.concat(
                                                        [beam_t[b],
                                                         tf.expand_dims(indices_t[bb], axis=1)
                                                         ], axis=1
                                                         )
                                                    )
                            c_state_candidates.append(c_state)
                            m_state_candidates.append(m_state)

                    temp_probs = tf.stack(probs_candidates, axis=1)
                    temp_indices = tf.stack(indices_candidates, axis=1)
                    temp_beam = tf.stack(beam_candidates, axis=1)
                    temp_c_states = tf.stack(c_state_candidates, axis=1)
                    temp_m_states = tf.stack(m_state_candidates, axis=1)
                    _, max_indices = tf.nn.top_k(temp_probs, k=config.beamsize, sorted=True)

                    #index
                    index = tf.add(
                                tf.matmul(b_index, be_index),
                                max_indices
                                )
                    prev_probs = tf.gather(tf.reshape(temp_probs, [-1]), index)
                    prev_indices = tf.gather(tf.reshape(temp_indices, [-1]), index)
                    beam = tf.gather(tf.reshape(temp_beam, [-1, time_index+1]), index)
                    prev_c_states = tf.gather(
                                            tf.reshape(
                                                temp_c_states,
                                                [-1, config.h_units]
                                            ),
                                            index
                                        )
                    prev_m_states = tf.gather(
                                            tf.reshape(
                                                temp_m_states,
                                                [-1, config.h_units]
                                            ),
                                            index
                                        )

            beam_t = tf.transpose(beam, [1,0,2])
            outputs = beam_t[0]

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

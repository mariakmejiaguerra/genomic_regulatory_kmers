# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' # Introduction
# ' Some of the code below is still under active development

# ' ## Required libraries
# + name = 'import_libraries', echo=False
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers.python.layers.layers import convolution2d
from tensorflow.contrib.layers.python.layers.layers import max_pool2d
from tensorflow.contrib.layers.python.layers.layers import fully_connected
from tensorflow.contrib.layers.python.layers.layers import flatten

class CNN:
    def __init__(self, sequence_length, kmer_sizes, num_classes, feature_number, random_int=1234):
        # Set a graph-level seed for reproducibility
        tf.set_random_seed(random_int)
        # TF graph
        with tf.variable_scope('placeholder'):
            # Variables created here will be named
            # "placeholder/input_x"
            self.input_x = tf.placeholder(tf.float32,
                                          [None, sequence_length * feature_number],
                                          name="input_x")
            # "placeholder/input_y"
            self.input_y = tf.placeholder(tf.float32,
                                          [None, num_classes],
                                          name="input_y")

            # "placeholder/dropout_keep_prob"
            self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                    name="dropout_keep_prob")

            # "placeholder/x_image"
            self.x_image = tf.reshape(self.input_x,
                                 [-1, sequence_length, feature_number, 1],
                                 name="input_image")  # one-hot encoded sequence

        with tf.variable_scope('network'):
            # Convolution Layers
            n_conv1 = 384

            l_conv1 = kmer_sizes[0]
            l_conv2 = kmer_sizes[1]
            l_conv3 = kmer_sizes[2]

            maxpool_len = 2

            filter_shape1 = [l_conv1, feature_number]

            n_steps1 = sequence_length - l_conv1 + 1

            conv1 = convolution2d(self.x_image,
                                  n_conv1,
                                  filter_shape1,
                                  padding='VALID',
                                  normalizer_fn=None)

            conv1_pool = max_pool2d(conv1,
                                    [maxpool_len, 1],
                                    [maxpool_len, 1])

            conv1_pool_len = int(n_steps1 / maxpool_len)

            n_steps2 = conv1_pool_len - l_conv2 + 1

            conv2 = convolution2d(conv1_pool,
                                  n_conv1,
                                  [l_conv2, 1],
                                  padding='VALID',
                                  normalizer_fn=None)

            conv2_pool = max_pool2d(conv2, [maxpool_len, 1], [maxpool_len, 1])

            conv2_pool_len = int(n_steps2 / maxpool_len)

            n_steps3 = conv2_pool_len - l_conv3 + 1

            conv3 = convolution2d(conv2_pool,
                                  n_conv1,
                                  [l_conv3, 1],
                                  padding='VALID',
                                  normalizer_fn=None)

            conv3_pool = max_pool2d(conv3, [maxpool_len, 1], [maxpool_len, 1])

            conv3_pool_len = int(n_steps3 / maxpool_len)

            final_maxpool_len = int(conv3_pool_len - l_conv3 + 1)  # final convolution layers
            final_conv = convolution2d(conv3_pool,
                                       n_conv1,
                                       [3, 1],
                                       padding='VALID',
                                       normalizer_fn=None)
            final_pool = max_pool2d(final_conv, [final_maxpool_len, 1], [final_maxpool_len, 1])

        # Output layers
        with tf.variable_scope("output"):
            self.scores = fully_connected(flatten(final_pool),
                                     num_classes,
                                     activation_fn=None)
            self.pred_softmax = tf.nn.softmax(self.scores,
                                         name="pred_softmax")

        # Define loss
        with tf.variable_scope("loss"):
            self.entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                   labels=self.input_y,
                                                                   name="entropy_loss")
            self.loss_op = tf.reduce_mean(self.entropy_loss,
                                     name="loss")

        # Accuracy
        with tf.variable_scope("accuracy"):
            self.correct_prediction = tf.equal(tf.argmax(self.input_y, 1),
                                          tf.argmax(self.pred_softmax, 1))
            self.accuracy_val = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32),
                                          name="accuracy")

        # Define a training procedure
        with tf.variable_scope('optimizer'):
            self.global_step = tf.Variable(0,
                                      name="global_step",
                                      trainable=False)
            self.increment_global_step = tf.assign_add(self.global_step, 1,
                                                  name='increment_global_step')  # Increment global step explicitly
            self.optimizer = tf.train.AdamOptimizer()
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                      name="train_op")

        # summaries for tensorboard reports
        grad_summaries = []
        for grad, var in self.grads_and_vars:
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name),
                                                         grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name),
                                                     tf.nn.zero_fraction(grad))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

        # summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss_op)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy_val)
        self.train_summary_op = tf.summary.merge([loss_summary,
                                             acc_summary,
                                             grad_summaries_merged],
                                            name="train_summary_op")
        self.eval_summary_op = tf.summary.merge([loss_summary,
                                            acc_summary],
                                           name="eval_summary_op")

        # define initialization of all variables
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

        self.saver = tf.train.Saver()



class RNN:
    def __init__(self, sequence_length, num_classes, feature_number, lstm_n_hidden, lstm_keep_prob, random_int=1234):
        tf.set_random_seed(random_int)
        with tf.variable_scope('placeholder'):
            # "placeholder/input_x"
            self.input_x = tf.placeholder(tf.float32,
                                     [None, sequence_length * feature_number],
                                     name="input_x")
            # "placeholder/input_y"
            self.input_y = tf.placeholder(tf.float32,
                                     [None, num_classes],
                                     name="input_y")

            # "placeholder/dropout_keep_prob"
            self.dropout_keep_prob = tf.placeholder(tf.float32,
                                               name="dropout_keep_prob")

            # "placeholder/x_image"
            self.x_image = tf.reshape(self.input_x,
                                 [-1, sequence_length, feature_number, 1],
                                 name="input_image")  # one-hot encoded sequence

        with tf.variable_scope('network'):
            x_unpacked = tf.unstack(self.x_image,
                                        axis=1)

            lstm_fw_cell = rnn_cell.BasicLSTMCell(lstm_n_hidden)

            lstm_bw_cell = rnn_cell.BasicLSTMCell(lstm_n_hidden)

            birnn_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                      lstm_bw_cell,
                                                                      x_unpacked,
                                                                      dtype=tf.float32)

            rnn_out = tf.div(tf.add_n(birnn_out),
                             sequence_length)

            rnn_out_drop = tf.nn.dropout(rnn_out,
                                         lstm_keep_prob)


        # Output layers
        with tf.variable_scope("output"):
            self.scores = fully_connected(flatten(rnn_out_drop),
                                     num_classes,
                                     activation_fn=None)
            self.pred_softmax = tf.nn.softmax(self.scores,
                                         name="pred_softmax")

        # Define loss
        with tf.variable_scope("loss"):
            self.entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                   labels=self.input_y,
                                                                   name="entropy_loss")
            self.loss_op = tf.reduce_mean(self.entropy_loss,
                                     name="loss")

        # Accuracy
        with tf.variable_scope("accuracy"):
            self.correct_prediction = tf.equal(tf.argmax(self.input_y, 1),
                                          tf.argmax(self.pred_softmax, 1))
            self.accuracy_val = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32),
                                          name="accuracy")

        # Define a training procedure
        with tf.variable_scope('optimizer'):
            self.global_step = tf.Variable(0,
                                      name="global_step",
                                      trainable=False)
            self.increment_global_step = tf.assign_add(self.global_step, 1,
                                                  name='increment_global_step')  # Increment global step explicitly
            self.optimizer = tf.train.AdamOptimizer()
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                 name="train_op")

        # summaries for tensorboard reports
        grad_summaries = []
        for grad, var in self.grads_and_vars:
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name),
                                                         grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name),
                                                     tf.nn.zero_fraction(grad))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

        # summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss_op)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy_val)
        self.train_summary_op = tf.summary.merge([loss_summary,
                                             acc_summary,
                                             grad_summaries_merged],
                                            name="train_summary_op")
        self.eval_summary_op = tf.summary.merge([loss_summary,
                                            acc_summary],
                                           name="eval_summary_op")

        # define initialization of all variables
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.saver = tf.train.Saver()


class CNN_LSTM:
    def __init__(self, sequence_length, kmer_sizes, num_classes, feature_number, rnn_n_hidden, rnn_keep_prob, random_int=1234):
        tf.set_random_seed(random_int)

        with tf.variable_scope('placeholder'):
            # "placeholder/input_x"
            self.input_x = tf.placeholder(tf.float32,
                                     [None, sequence_length * feature_number],
                                     name="input_x")
            # "placeholder/input_y"
            self.input_y = tf.placeholder(tf.float32,
                                     [None, num_classes],
                                     name="input_y")

            # "placeholder/dropout_keep_prob"
            self.dropout_keep_prob = tf.placeholder(tf.float32,
                                               name="dropout_keep_prob")

            # "placeholder/x_image"
            self.x_image = tf.reshape(self.input_x,
                                 [-1, sequence_length, feature_number, 1],
                                 name="input_image")  # Sequence like image

        with tf.variable_scope('network'):
            # Convolution Layer
            n_conv1 = 128
            l_conv1 = kmer_sizes[0]
            filter_shape = [l_conv1, feature_number]  # dense
            n_steps1 = (sequence_length - kmer_sizes[0] + 1)

            conv = convolution2d(self.x_image,
                                 n_conv1,
                                 filter_shape,
                                 padding='VALID',
                                 normalizer_fn=None)

            conv_resh = tf.reshape(conv,
                                   [-1, n_steps1, n_conv1],
                                   name="reshape")

            conv_unstacked = tf.unstack(conv_resh,
                                        axis=1)

            lstm_fw_cell = rnn_cell.BasicLSTMCell(rnn_n_hidden)
            lstm_bw_cell = rnn_cell.BasicLSTMCell(rnn_n_hidden)
            birnn_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                      lstm_bw_cell,
                                                                      conv_unstacked,
                                                                      dtype=tf.float32)
            rnn_out = tf.div(tf.add_n(birnn_out),
                             sequence_length)

            rnn_out_drop = tf.nn.dropout(rnn_out,
                                         rnn_keep_prob)

        # Output layers
        with tf.variable_scope("output"):
            self.scores = fully_connected(flatten(rnn_out_drop),
                                     num_classes,
                                     activation_fn=None)
            self.pred_softmax = tf.nn.softmax(self.scores,
                                         name="pred_softmax")

        # Define loss
        with tf.variable_scope("loss"):
            self.entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                   labels=self.input_y,
                                                                   name="entropy_loss")
            self.loss_op = tf.reduce_mean(self.entropy_loss,
                                     name="loss")

        # Accuracy
        with tf.variable_scope("accuracy"):
            self.correct_prediction = tf.equal(tf.argmax(self.input_y, 1),
                                          tf.argmax(self.pred_softmax, 1))
            self.accuracy_val = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32),
                                          name="accuracy")

        # Define a training procedure
        with tf.variable_scope('optimizer'):
            self.global_step = tf.Variable(0,
                                      name="global_step",
                                      trainable=False)
            self.increment_global_step = tf.assign_add(self.global_step, 1,
                                                  name='increment_global_step')  # Increment global step explicitly
            self.optimizer = tf.train.AdamOptimizer()
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                 name="train_op")

        # summaries for tensorboard reports
        grad_summaries = []
        for grad, var in self.grads_and_vars:
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name),
                                                         grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name),
                                                     tf.nn.zero_fraction(grad))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

        # summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss_op)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy_val)
        self.train_summary_op = tf.summary.merge([loss_summary,
                                             acc_summary,
                                             grad_summaries_merged],
                                            name="train_summary_op")
        self.eval_summary_op = tf.summary.merge([loss_summary,
                                            acc_summary],
                                           name="eval_summary_op")

        # define initialization of all variables
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.saver = tf.train.Saver()


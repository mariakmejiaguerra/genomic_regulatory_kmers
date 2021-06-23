# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' # Introduction
# ' Some of the code below is still under active development

# ' ## Required libraries
# + name = 'import_libraries', echo=False
import sys
import os
import numpy as np

from kmergrammar.kgutils.data_utils import get_features_and_labels
from kmergrammar.kgutils.data_utils import data_in_dir
from kmergrammar.kgutils.data_utils import batch_index
from kmergrammar.kgutils.data_utils import save_plot_roc
from kmergrammar.kgutils.data_utils import evaluate_prediction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# + name= 'load_dataset', echo=False
def load_test_dataset(logger, dataset_num, data_files, batch_size=128, train=False):
    global _test_step
    global _test_epochs
    global _iterate_test
    global _test_features
    global _test_labels
    global _test_size
    global _sequence_length

    dataset_name = _datasets[dataset_num]
    if train:
        print("FATAL! method exclusively for test data")
        sys.exit(1)
    else:
        _test_features, _test_labels, _sequence_length = get_features_and_labels(dataset_name,
                                                                                 data_files,
                                                                                 logger,
                                                                                 train=False)
        print("sequence length: {}".format(_sequence_length))
        _test_size = int(_test_features.shape[0])
        _test_step = 0
        _test_epochs = 0
        _iterate_test = range(0, (1 + int(len(_test_labels) / batch_size)))
        print("total examples in test set: {}".format(len(_test_labels) ))
        print("total batches to go through the test set: {}".format(int(len(_test_labels) / batch_size)))


# + name= 'test_batch_iter', echo=False
def test_batch_iter(current_step, dataset_to_use=2, batch_size=128):
    global _test_features
    global _test_labels
    global _iterate_test

    if dataset_to_use == 0:
        print("FATAL! method exclusively for test data only")
        sys.exit(1)
    elif dataset_to_use == 1:
        print("FATAL! method exclusively for test data only")
        sys.exit(1)
    elif dataset_to_use == 2:
        data_size = len(_test_labels)
        index_batches = batch_index(range(0, len(_test_labels) + 1), batch_size)
        start = index_batches[current_step][0]
        end = index_batches[current_step][-1]
        first_step = _iterate_test[0]

    if current_step == first_step:  # starting iterations
        shuffle_indices = np.random.permutation(np.arange(data_size))
        if dataset_to_use == 0:
            print("FATAL! method exclusively for test data only")
            sys.exit(1)
        elif dataset_to_use == 1:
            print("FATAL! method exclusively for test data only")
            sys.exit(1)
        elif dataset_to_use == 2:
            _test_features = _test_features[shuffle_indices]
            _test_labels = _test_labels[shuffle_indices]
            return _test_features[start:end], _test_labels[start:end]
    else:
        if dataset_to_use == 0:
            print("FATAL! method exclusively for test data only")
            sys.exit(1)
        elif dataset_to_use == 1:
            print("FATAL! method exclusively for test data only")
            sys.exit(1)
        elif dataset_to_use == 2:
            return _test_features[start:end], _test_labels[start:end]


def write_test_file(test_file, test_params):
    with open(test_file, "w") as text_file:
        text_file.write("input_path\t" + test_params[0]+"\n")
        text_file.write("dataset_name\t"+test_params[1]+"\n")
        text_file.write("base_model_dir\t" + test_params[2]+"\n")
        text_file.write("model_name\t"+test_params[3]+"\n")
        text_file.write("model_type\t"+test_params[4]+"\n")


# + name= 'prediction_test', echo=False
def prediction_test(sess, test=True, prev_auc=0.0001):
    if test:
        # bring what we need to make predictions on test data
        global_step = sess.graph.get_tensor_by_name('optimizer/global_step:0')
        input_x = sess.graph.get_tensor_by_name('placeholder/input_x:0')
        input_y = sess.graph.get_tensor_by_name('placeholder/input_y:0')
        dropout_keep_prob = sess.graph.get_tensor_by_name('placeholder/dropout_keep_prob:0')
        pred_softmax = sess.graph.get_tensor_by_name('output/pred_softmax:0')
        eval_summary_op = sess.graph.get_tensor_by_name('eval_summary_op/eval_summary_op:0')
        test_operations = [global_step, eval_summary_op, pred_softmax]

        # actual prediction happening here!
        pred_test_labels = None
        true_test_labels = None
        for test_step in _iterate_test:
            test_batch = test_batch_iter(test_step,
                                    dataset_to_use=2,
                                    batch_size=128)
            if true_test_labels is None:
                feed_dict = {
                    input_x: test_batch[0],
                    input_y: test_batch[1],
                    dropout_keep_prob: 1.0}
                gs, summ, preds = sess.run(test_operations, feed_dict)
                pred_test_labels = preds
                true_test_labels = test_batch[1]
            else:
                feed_dict = {
                    input_x: test_batch[0],
                    input_y: test_batch[1],
                    dropout_keep_prob: 1.0}
                gs, summ, preds = sess.run(test_operations, feed_dict)
                pred_test_labels = np.vstack([pred_test_labels, preds])
                true_test_labels = np.vstack([true_test_labels, test_batch[1]])

        # collect a dictionary with fpr, accuracy, precision, etc.,
        evaluation_stats = evaluate_prediction(true_test_labels,
                                               pred_test_labels,
                                               prev_auc)
    else:
        print("FATAL! method exclusively for test data only")
        sys.exit(1)
    return evaluation_stats


# + name= 'test_from_restored_model', echo=False
def test_from_restored_model(arguments, logger, run_id):
    global _datasets
    working_dir, data_unit = data_in_dir(arguments.input_path, arguments.dataset_name)
    _datasets = data_unit
    logger.info("WORKING_DIR")
    logger.info(working_dir)
    logger.info("PARAMETERS:")

    for dataset_num, regulator in enumerate(_datasets):
        logger.info("input: dataset idx")
        logger.info(str(dataset_num))
        logger.info("input: dataset name")
        logger.info(str(regulator))

        data_files = {"database": os.path.join(working_dir, ''.join([regulator, '.db']))}

        print("-" * 80)
        print("Restoring existing {} model".format(arguments.model_type))
        print("Reading data from folder {} and database {}".format(arguments.dataset_name, str(regulator)))

        # load meta graph to restore final model
        logger.info("kmergrammar testing model: {}".format(arguments.model_name))
        latest_checkpoint = tf.train.latest_checkpoint(arguments.base_model_dir + arguments.model_name + "/checkpoints/")
        logger.info("kmergrammar restore model_final from: {}".format(latest_checkpoint + ".meta"))

        print("Loading test table...")
        load_test_dataset(logger,
                     dataset_num,
                     data_files,
                     train=False)

        # import the graph from the file
        imported_graph = tf.train.import_meta_graph(latest_checkpoint + ".meta",
                                                    clear_devices=True)
        # establish the init operation
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # run the session
        with tf.Session() as sess:
            # actual initialization
            sess.run(init_op)

            # restore the saved variables
            imported_graph.restore(sess, latest_checkpoint)

            # uncomment next block of code for debugging purposes
            """
            logger.info("")
            logger.info("Extracting variables from model\n")
            for var in sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                logger.info(var.name)

            logger.info("")
            logger.info("Extracting operations from model\n")
            for op in sess.graph.get_operations():
                logger.info(op.name)
            """

            test_stats = prediction_test(sess, test=True)
            eval_summary_dir = arguments.base_model_dir + arguments.model_name + "/summaries/eval"
            plot_file = eval_summary_dir + "/plot_roc_restored_test." + arguments.model_name + ".png"
            save_plot_roc(test_stats.get("fpr"), test_stats.get("tpr"), test_stats.get("roc_auc"), plot_file, arguments.model_name)
            print("{}, dataset {}, final test roc auc {}".format(_datasets[dataset_num], dataset_num, test_stats.get("roc_auc")))


# + name= 'test_from_current_session', echo=False
def test_from_running_session(sess, logger, dataset_num, data_files, description):
    print("Loading test table...")
    load_test_dataset(logger,
                 dataset_num,
                 data_files,
                 train=False)

    evaluate_dict = prediction_test(sess, test=True)
    print("{}, dataset {}, final test roc auc {}".format(_datasets[dataset_num],
                                                         dataset_num,
                                                         evaluate_dict.get("roc_auc")))
    save_plot_roc(evaluate_dict.get("fpr"),
                  evaluate_dict.get("tpr"),
                  evaluate_dict.get("roc_auc"),
                  data_files.get("test_plot_file"), description)


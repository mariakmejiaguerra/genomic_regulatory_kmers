# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' # Introduction
# ' Some of the code below is still under active development

# ' ## Required libraries
# + name = 'import_libraries', echo=False
import sys
import os
import inspect
import numpy as np

from kmergrammar.kgutils.data_utils import get_features_and_labels
from kmergrammar.kgutils.data_utils import data_in_dir
from kmergrammar.kgutils.data_utils import batch_index
from kmergrammar.kgutils.data_utils import save_validation_plot_roc
from kmergrammar.kgutils.data_utils import evaluate_prediction

from kmergrammar.kgmodel.kgdeep.architectures import CNN, RNN, CNN_LSTM
from kmergrammar.kgmodel.kgdeep.test import test_from_running_session
from kmergrammar.kgmodel.kgdeep.test import write_test_file

from sklearn import model_selection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.contrib import slim


# + name= 'model_summary', echo=False
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# + name= 'prediction', echo=False
def prediction_validation(sess, validate=True, prev_auc=0.0001):
    if validate:
        # bring what we need to make predictions on validation data
        input_x = sess.graph.get_tensor_by_name('placeholder/input_x:0')
        input_y = sess.graph.get_tensor_by_name('placeholder/input_y:0')
        dropout_keep_prob = sess.graph.get_tensor_by_name('placeholder/dropout_keep_prob:0')
        pred_softmax = sess.graph.get_tensor_by_name('output/pred_softmax:0')
        eval_summary_op = sess.graph.get_tensor_by_name('eval_summary_op/eval_summary_op:0')
        val_operations = [pred_softmax, eval_summary_op]

        # actual prediction happening here!
        pred_validation_labels = None
        true_validation_labels = None
        for validation_step in _iterate_validation:
            validation_batch = batch_iter(validation_step,
                                          dataset_to_use=1,
                                          batch_size=128)
            if pred_validation_labels is None:
                feed_dict = {
                    input_x: validation_batch[0],
                    input_y: validation_batch[1],
                    dropout_keep_prob: 1.0}
                preds, v_summaries = sess.run(val_operations, feed_dict)
                pred_validation_labels = preds
                true_validation_labels = validation_batch[1]
            else:
                feed_dict = {
                    input_x: validation_batch[0],
                    input_y: validation_batch[1],
                    dropout_keep_prob: 1.0}
                preds, v_summaries = sess.run(val_operations, feed_dict)
                pred_validation_labels = np.vstack([pred_validation_labels,
                                                    preds])
                true_validation_labels = np.vstack([true_validation_labels,
                                                    validation_batch[1]])

        # collect a dictionary with fpr, accuracy, precision, etc.,
        evaluation_stats = evaluate_prediction(true_validation_labels,
                                               pred_validation_labels,
                                               prev_auc)
        evaluation_stats["summary"] = v_summaries
    return evaluation_stats


# + name= 'load_dataset', echo=False
def load_dataset(logger, dataset_num, data_files, percvalidation, batch_size=128, train=True):
    global _train_step
    global _train_epochs
    global _iterate_train
    global _iterate_validation
    global _train_features
    global _validation_features
    global _train_labels
    global _validation_labels
    global _validation_size
    global _sequence_length

    dataset_name = _datasets[dataset_num]
    if train:
        features, labels, _sequence_length = get_features_and_labels(dataset_name,
                                                                     data_files,
                                                                     logger,
                                                                     train=True)
        validation_size = int(features.shape[
                              0] * percvalidation)
        print("sequence length: {}".format(_sequence_length))
        np.random.seed(0)
        random_seed = np.random.randint(42, size=1)[0]
        _train_features, _validation_features, _train_labels, _validation_labels = model_selection.train_test_split(
            features,
            labels,
            test_size=validation_size,
            random_state=random_seed)
        check_dataset_size(_train_labels, percvalidation)
        print("train/validation split: {:d}/{:d}".format(len(_train_labels),
                                                         len(_validation_labels)))

        _train_step = 0
        _train_epochs = 0
        _validation_size = validation_size
        _iterate_train = range(0, (1 + int(len(_train_labels) / batch_size)))
        _iterate_validation = range(0, (1 + int(len(_validation_labels) / batch_size)))
        print("total batches to go through the training set: {}".format(int(len(_train_labels) / batch_size)))
        print("total batches to go through the validation set: {}".format(int(len(_validation_labels) / batch_size)))
    else:
        print("FATAL! method exclusively for test data")
        sys.exit(1)


# + name= 'batch_iter', echo=False
def batch_iter(current_step, dataset_to_use=0, batch_size=128):
    global _train_step
    global _train_epochs
    global _train_features
    global _train_labels
    global _iterate_train
    global _validation_features
    global _validation_labels
    global _iterate_validation

    if dataset_to_use == 0:
        data_size = len(_train_labels)
        index_batches = batch_index(range(0, len(_train_labels) + 1), batch_size)
        start = index_batches[current_step][0]
        end = index_batches[current_step][-1]
        first_step = _iterate_train[0]
    elif dataset_to_use == 1:
        data_size = len(_validation_labels)
        index_batches = batch_index(range(0, len(_validation_labels) + 1), batch_size)
        start = index_batches[current_step][0]
        end = index_batches[current_step][-1]
        first_step = _iterate_validation[0]

    if current_step == first_step:  # starting iterations
        shuffle_indices = np.random.permutation(np.arange(data_size))
        if dataset_to_use == 0:
            _train_features = _train_features[shuffle_indices]
            _train_labels = _train_labels[shuffle_indices]
            _train_step += 1
            return _train_features[start:end], _train_labels[start:end]
        elif dataset_to_use == 1:
            _validation_features = _validation_features[shuffle_indices]
            _validation_labels = _validation_labels[shuffle_indices]
            return _validation_features[start:end], _validation_labels[start:end]
    else:
        if dataset_to_use == 0:
            if current_step < _iterate_train[-1]:
                _train_step += 1
                return _train_features[start:end], _train_labels[start:end]
            elif current_step == _iterate_train[-1]:
                _train_step = 0
                _train_epochs += 1
                return _train_features[start:end], _train_labels[start:end]
        elif dataset_to_use == 1:
            return _validation_features[start:end], _validation_labels[start:end]


# + name= 'check_dataset_size', echo=False
def check_dataset_size(labels, percvalidation):
    global _notenoughtotrain
    classes, counts = np.unique(labels, return_counts=True)
    print("got {:d} examples for class {:0.1f} and {:d} examples for class {:0.1f}".format(counts[0], classes[0],
                                                                                           counts[1], classes[1]))

    # determine if we have enough observations to train a model
    perctrain = 1 - percvalidation
    train_size = counts[1] * perctrain
    if int(train_size // 1) < 10:
        _notenoughtotrain = True
    else:
        _notenoughtotrain = False


# + name= 'train_model', echo=False
def train_new_model(arguments, logger, run_id):
    global _datasets
    # arguments are tensorflow flags
    arguments._parse_flags()

    model_type = arguments.model_type
    working_dir, _datasets = data_in_dir(arguments.input_path, arguments.dataset_name)

    logger.info("WORKING_DIR")
    logger.info(working_dir)
    logger.info("PARAMETERS:")

    for attr, value in sorted(arguments.__flags.items()):
        logger.info("{}={}".format(attr.upper(),
                                   value))

    for dataset_num, regulator in enumerate(_datasets):
        logger.info("input: dataset idx")
        logger.info(str(dataset_num))
        logger.info("input: dataset name")
        logger.info(str(regulator))

        data_files = {"database": os.path.join(working_dir, ''.join([regulator, '.db']))}

        print("-" * 80)
        print("Training a {} model".format(model_type))
        print("Folder %s - database %s" % (arguments.dataset_name, str(regulator)))
        print("Loading train table...")
        load_dataset(logger, dataset_num, data_files, arguments.percvalidation)

        logger.info("input: percentage for validation")
        logger.info(str(arguments.percvalidation))

        # Root directory for models and summaries
        kgresults = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
            "kgmodel/kgdeep", "kgresults")
        if not os.path.exists(kgresults):
            os.makedirs(kgresults)

        if _notenoughtotrain:
            print("not enough observations to train a model")
            logger.info("kmergrammar did not train the model {}".format(run_description))
            continue
        else:
            run_id_model_id = "{}.{}".format(run_id,
                                             str(dataset_num))
            run_description = "{}.{}.k{}.{}".format(arguments.dataset_name,
                                                    model_type,
                                                    arguments.kmer_sizes.replace(",", "k"),
                                                    run_id_model_id)
            logger.info("kmergrammar training model: {}".format(run_description))

            # Current directory for models and summaries
            out_dir = os.path.abspath(os.path.join(kgresults, "runs", run_description))
            print("Writing models to folder {}\n".format(os.path.abspath(out_dir)))

            # Directories for summaries and checkpoints should exists
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            eval_summary_dir = os.path.join(out_dir, "summaries", "eval")
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Files for validation and test plots
            val_plot_file = eval_summary_dir + "/plot_roc_validation." + run_description + ".png"
            test_plot_file = eval_summary_dir + "/plot_roc_test." + run_description + ".png"

            # Prepare param file for reuse of the model
            test_file = os.path.join(working_dir, ''.join(["restore_model.", run_description, ".txt"]))

            # Add to data_file dictionaries
            data_files["val_plot_file"] = val_plot_file
            data_files["test_plot_file"] = test_plot_file
            data_files["test_param_file"] = test_file

            if arguments.model_type == "CNN":
                network = CNN(_sequence_length,
                              list(map(int, arguments.kmer_sizes.split(","))),
                              arguments.classes_number,
                              arguments.feature_number,
                              random_int=1234)

            elif arguments.model_type == "CNN_LSTM":
                network = CNN_LSTM(_sequence_length,
                                   list(map(int, arguments.kmer_sizes.split(","))),
                                   arguments.classes_number,
                                   arguments.feature_number,
                                   arguments.rnn_n_hidden,
                                   arguments.rnn_keep_prob,
                                   random_int=2345)

            elif arguments.model_type == "LSTM":
                network = RNN(_sequence_length,
                              arguments.classes_number,
                              arguments.feature_number,
                              arguments.rnn_n_hidden,
                              arguments.rnn_keep_prob,
                              random_int=3456)

            session_conf = tf.ConfigProto(
                allow_soft_placement=arguments.allow_soft_placement,
                log_device_placement=arguments.log_device_placement)

            with tf.Session(config=session_conf) as sess:
                # train and validation summaries
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
                eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

                # actual initialization
                sess.run(network.init_op)

                # uncomment next line for debugging!
                # model_summary()

                prev_auc = 0.0001  # it prevents DIV0
                stop_condition = None
                evaluations = []
                while stop_condition is None:
                    training = [network.train_op,
                                network.increment_global_step,
                                network.train_summary_op,
                                network.loss_op,
                                network.accuracy_val]

                    # BTW, train_step is an integer and globally updated
                    train_batch = batch_iter(_train_step,
                                             dataset_to_use=0,
                                             batch_size=arguments.batch_size)
                    feed_dict = {
                        network.input_x: train_batch[0],
                        network.input_y: train_batch[1],
                        network.dropout_keep_prob: arguments.dropout_keep_prob}

                    _, current_step, t_summaries, loss, acc = sess.run(training, feed_dict)
                    train_summary_writer.add_summary(t_summaries,
                                                     current_step)

                    logger.info("training: step {}, loss {:g}, acc {:g}".format(sess.run(network.global_step),
                                                                                loss,
                                                                                acc))

                    if sess.run(network.global_step) % arguments.evaluate_every == 0:
                        network.saver.save(sess, checkpoint_prefix, global_step=network.global_step)
                        validation_stats = prediction_validation(sess, prev_auc=prev_auc)
                        validation_report = "validation: train epoch {}, train step {}, " \
                                            "accuracy {:g}, avg_precision {:0.2f}, prev_roc_auc {:0.4f}, " \
                                            "roc_auc {:0.4f}, auc delta {:0.4f}".format(str(_train_epochs),
                                                                                        str(current_step),
                                                                                        validation_stats.get("accuracy"),
                                                                                        validation_stats.get("avg_prec"),
                                                                                        prev_auc,
                                                                                        validation_stats.get("roc_auc"),
                                                                                        validation_stats.get("perc_chg_auc"))
                        prev_auc = validation_stats.get("roc_auc")
                        validation_stats["report"] = validation_report
                        evaluations.append(validation_stats)
                        eval_summary_writer.add_summary(validation_stats.get("summary"),
                                                        current_step)
                        print(validation_report)

                    if _train_epochs == arguments.num_epochs:
                        stop_condition = 1

                    # if early-stop, get delta auc for the last 1000 evaluations on validation
                    if arguments.early_stop and len(evaluations) >= 1000:
                        delta_on_val = [d["perc_chg_auc"] for d in evaluations[-1000:]]
                        # if for more than half of the evaluation steps the model doesn't seem to improve - stop
                        if len([item for item in delta_on_val if item <= 0]) > 500:
                            stop_condition = 1

                # get results on validation for the training session
                fpr_val = [d["fpr"] for d in evaluations]
                tpr_val = [d["tpr"] for d in evaluations]
                aucs_val = [d["roc_auc"] for d in evaluations]

                for d in evaluations:
                    logger.info(d.get("report"))

                # plot results on validation for the training session
                save_validation_plot_roc(fpr_val,
                                         tpr_val,
                                         aucs_val,
                                         data_files.get("val_plot_file"),
                                         run_description)

                # save final model
                model_final = os.path.join(checkpoint_dir, "model_final")
                path = network.saver.save(sess, model_final, global_step=network.global_step)
                print("Saved final model checkpoint to {}\n".format(path))


                test_params = [arguments.input_path,
                                   arguments.dataset_name,
                                   os.path.abspath(out_dir).replace(run_description, ""),
                                   run_description,
                                   arguments.model_type]
                write_test_file(data_files.get("test_param_file"), test_params)
                print("To restore the model use the following parameter file {}\n".format(
                    data_files.get("test_param_file")))
                if arguments.test_and_train:
                    test_from_running_session(sess,
                                              logger,
                                              dataset_num,
                                              data_files,
                                              run_description)



# + name= 'explore_graph', echo=False
def explore_restored_graph(graph, logger):
    """
    # to write all the variables and operations in the graph to the log file
    :param graph:
    :param logger:
    :return:
    """
    logger.info("")
    logger.info("Extracting variables from model\n")
    for var in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        logger.info(var.name)

    logger.info("")
    logger.info("Extracting operations from model\n")
    for op in graph.get_operations():
        logger.info(op.name)


# + name= 'training_from_restored_model', echo=False
def restore_training(arguments, logger, run_id):
    global _datasets
    model_type = arguments.model_type
    working_dir, data_unit = data_in_dir(arguments.input_path, arguments.dataset_name)
    _datasets = data_unit
    logger.info("WORKING_DIR")
    logger.info(working_dir)
    logger.info("PARAMETERS:")

    for attr, value in sorted(arguments.__flags.items()):
        logger.info("{}={}".format(attr.upper(),
                                   value))

    for dataset_num, regulator in enumerate(_datasets):
        logger.info("input: dataset idx")
        logger.info(str(dataset_num))
        logger.info("input: dataset name")
        logger.info(str(regulator))

        data_files = {"database": os.path.join(working_dir, ''.join([regulator, '.db']))}

        # load meta graph to restore final model
        logger.info("kmergrammar restart {} model for training: {}".format(arguments.model_type,
                                                                           arguments.model_name))

        latest_checkpoint = tf.train.latest_checkpoint(
            arguments.base_model_dir + arguments.model_name + "/checkpoints/")
        logger.info("kmergrammar restored model_final from: {}".format(latest_checkpoint + ".meta"))

        print("-" * 80)
        print("Restoring training a {} model".format(model_type))
        print("Folder %s - database %s" % (arguments.dataset_name, str(regulator)))
        print("Loading train table...")
        load_dataset(logger,
                     dataset_num,
                     data_files,
                     arguments.percvalidation)

        logger.info("input: percentage for validation")
        logger.info(str(arguments.percvalidation))

        # Root directory for models and summaries
        kgresults = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
            "kgmodel/kgdeep", "kgresults")
        if not os.path.exists(kgresults):
            os.makedirs(kgresults)

        run_id_model_id = "{}.{}".format(run_id,
                                         str(dataset_num))
        run_description = "{}.{}.k{}.{}".format(arguments.dataset_name,
                                                model_type,
                                                arguments.kmer_sizes.replace(",", "k"),
                                                run_id_model_id)
        logger.info("kmergrammar training model: {}".format(run_description))

        # Current directory for models and summaries
        out_dir = os.path.abspath(os.path.join(kgresults, "runs", run_description))
        print("Writing models to folder {}\n".format(os.path.abspath(out_dir)))

        # Directories for summaries and checkpoints should exists
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        eval_summary_dir = os.path.join(out_dir, "summaries", "eval")
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Files for validation and test plots
        val_plot_file = eval_summary_dir + "/plot_restored_roc_validation." + run_description + ".png"
        test_plot_file = eval_summary_dir + "/plot_restored_roc_test." + run_description + ".png"

        # Prepare param file for reuse of the model
        test_file = os.path.join(working_dir, ''.join(["restore_model.", run_description, ".txt"]))

        # Add to data_file dictionaries
        data_files["val_plot_file"] = val_plot_file
        data_files["test_plot_file"] = test_plot_file
        data_files["test_param_file"] = test_file

        # import the graph from the file
        new_saver = tf.train.import_meta_graph(latest_checkpoint + ".meta",
                                                    clear_devices=True)
        # establish the init operation
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # run the session
        with tf.Session() as sess:
            # train and validation summaries
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

            # actual initialization
            sess.run(init_op)

            # restore the saved variables
            new_saver.restore(sess, latest_checkpoint)

            # uncomment next line only for debugging purposes
            # explore_restored_graph(sess.graph, logger)

            # bring what we need to continue training the model
            network_global_step = sess.graph.get_tensor_by_name('optimizer/global_step:0')
            network_input_x = sess.graph.get_tensor_by_name('placeholder/input_x:0')
            network_input_y = sess.graph.get_tensor_by_name('placeholder/input_y:0')
            network_dropout_keep_prob = sess.graph.get_tensor_by_name('placeholder/dropout_keep_prob:0')

            network_train_op = sess.graph.get_tensor_by_name('eval_summary_op/eval_summary_op:0')
            network_increment_global_step = sess.graph.get_tensor_by_name('eval_summary_op/eval_summary_op:0')
            network_train_summary_op = sess.graph.get_tensor_by_name('eval_summary_op/eval_summary_op:0')
            network_loss_op = sess.graph.get_tensor_by_name('eval_summary_op/eval_summary_op:0')
            network_accuracy_val = sess.graph.get_tensor_by_name('eval_summary_op/eval_summary_op:0')

            prev_auc = 0.0001  # it prevents DIV0
            stop_condition = None
            evaluations = []
            while stop_condition is None:
                training = [network_train_op,
                            network_increment_global_step,
                            network_train_summary_op,
                            network_loss_op,
                            network_accuracy_val]

                # BTW, train_step is an integer and globally updated
                train_batch = batch_iter(_train_step,
                                         dataset_to_use=0,
                                         batch_size=arguments.batch_size)
                feed_dict = {
                    network_input_x: train_batch[0],
                    network_input_y: train_batch[1],
                    network_dropout_keep_prob: arguments.dropout_keep_prob}

                _, current_step, t_summaries, loss, acc = sess.run(training, feed_dict)
                train_summary_writer.add_summary(t_summaries,
                                                 current_step)

                logger.info("training: step {}, loss {:g}, acc {:g}".format(sess.run(network_global_step),
                                                                            loss,
                                                                            acc))

                if sess.run(network_global_step) % arguments.evaluate_every == 0:
                    new_saver.save(sess, checkpoint_prefix, global_step=network_global_step)
                    validation_stats = prediction_validation(sess, prev_auc=prev_auc)
                    validation_report = "validation: train epoch {}, train step {}, " \
                                        "accuracy {:g}, avg_precision {:0.2f}, prev_roc_auc {:0.4f}, " \
                                        "roc_auc {:0.4f}, auc delta {:0.4f}".format(str(_train_epochs),
                                                                                    str(current_step),
                                                                                    validation_stats.get("accuracy"),
                                                                                    validation_stats.get("avg_prec"),
                                                                                    prev_auc,
                                                                                    validation_stats.get("roc_auc"),
                                                                                    validation_stats.get(
                                                                                        "perc_chg_auc"))
                    prev_auc = validation_stats.get("roc_auc")
                    validation_stats["report"] = validation_report
                    evaluations.append(validation_stats)
                    eval_summary_writer.add_summary(validation_stats.get("summary"),
                                                    current_step)
                    print(validation_report)

                if _train_epochs == arguments.num_epochs:
                    stop_condition = 1

                # if early-stop, get delta auc for the last 1000 evaluations on validation
                if arguments.early_stop and len(evaluations) >= 1000:
                    delta_on_val = [d["perc_chg_auc"] for d in evaluations[-1000:]]
                    # if for more than half of the evaluation steps the model doesn't seem to improve - stop
                    if len([item for item in delta_on_val if item <= 0]) > 500:
                        stop_condition = 1

            # get results on validation for the training session
            fpr_val = [d["fpr"] for d in evaluations]
            tpr_val = [d["tpr"] for d in evaluations]
            aucs_val = [d["roc_auc"] for d in evaluations]

            for d in evaluations:
                logger.info(d.get("report"))

            # plot results on validation for the training session
            save_validation_plot_roc(fpr_val,
                                     tpr_val,
                                     aucs_val,
                                     data_files.get("val_plot_file"),
                                     run_description)

            # save final model
            model_final = os.path.join(checkpoint_dir, "model_final")
            path = new_saver.save(sess, model_final, global_step=network_global_step)
            print("Saved final model checkpoint to {}\n".format(path))

            if arguments.test_and_train:
                test_params = [arguments.input_path,
                               arguments.dataset_name,
                               os.path.abspath(out_dir).replace(run_description, ""),
                               run_description]
                write_test_file(data_files.get("test_param_file"), test_params)
                print(
                    "To test the model use the following parameter file {}\n".format(data_files.get("test_param_file")))
                test_from_running_session(sess,
                                          logger,
                                          dataset_num,
                                          data_files,
                                          run_description)

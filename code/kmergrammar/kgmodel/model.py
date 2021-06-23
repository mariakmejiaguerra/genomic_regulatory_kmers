# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' # Introduction
# ' Some of the code below is under active development

# ' ## Required libraries
# + name = 'import_libraries', echo=False
import sys
import pandas as pd
from collections import namedtuple

from .kgdeep.train import train_new_model
from .kgdeep.test import test_from_restored_model

# + name = 'sort_model', echo=False
def sort_model(arguments, logger, run_id):
    """
    Parameters
    ----------
    Return
    ----------
    """
    if arguments.action == "train":
        model_params = get_train_parameters(arguments.paramfile,
                                            arguments.type)
        train_new_model(model_params,
                        logger,
                        run_id)

    elif arguments.action == "test":
        model_params = process_param_file(arguments.paramfile)

        test_from_restored_model(model_params,
                                 logger,
                                 run_id)


# + name = 'get_train_parameters', echo=False
def process_param_file(paramfile):
    """
    Parameters
    ----------
    Return
    ----------
    """
    df = pd.read_csv(paramfile, sep="\t", header=None, usecols=[0, 1], names=["keys", "value"])
    keywords = [key for key in df["keys"]]
    values = [val for val in df["value"]]
    model_args = namedtuple("model_args", keywords)
    return model_args(*values)


# + name = 'get_train_parameters', echo=False
def get_train_parameters(paramfile, type):
    import tensorflow as tf
    import math
    """
    Parameters
    ----------

    Return
    ----------
    """
    if type == "vkmer":
        print("Hola")
    elif type == "bagofkmers":
        print("Hello")
    elif type == "RNN":
        print("Laphi!")
    elif type == "CNN":
        print("Salve amicus")

        input_arguments = process_param_file(paramfile)

        required_fields = ["dataset_name",
                           "input_path",
                           "kmer_sizes"]

        missing_fields = [req for req in required_fields if req not in input_arguments._fields]

        if len(missing_fields) > 0:
            print("FATAL! missing required parameters: " + missing_fields)
            sys.exit(1)
        else:
            input_k = list(map(int, input_arguments.kmer_sizes.split(",")))
            if len(input_k) != 3:
                print("FATAL! wrong kmer_sizes parameter: " + input_arguments.kmer_sizes)
                print("input three integers separated by comma e.g, 9,7,5")
                sys.exit(1)

        # Data loading params
        tf.flags.DEFINE_string("model_type",
                               type,
                               "Type of model.")

        if "dataset_name" in input_arguments._fields:
            tf.flags.DEFINE_string("dataset_name",
                                   input_arguments.dataset_name,
                                   "Name for the data.")
        else:
            print("Provide a dataset_name")
            sys.exit(1)

        if "input_path" in input_arguments._fields:
            tf.flags.DEFINE_string("input_path",
                                   input_arguments.input_path,
                                   "Path for the data.")

        # Misc Parameters
        tf.flags.DEFINE_boolean("allow_soft_placement",
                                True,
                                "Allow device soft device placement (Default: True)")
        tf.flags.DEFINE_boolean("log_device_placement",
                                False,
                                "Log placement of ops on devices (Default: False)")
        tf.flags.DEFINE_boolean("fail_train_size",
                                False,
                                "Default allows to train a model (Default: False)")

        tf.flags.DEFINE_boolean("test_and_train",
                                False,
                                "Proceed to test after training (Default: False)")

        # Model hyperparameters
        tf.flags.DEFINE_integer("feature_number",
                                4,
                                "expected input are strings with valid alphabet ATCG, " +
                                "anything else is assumed to be N (Default: 4)")

        tf.flags.DEFINE_integer("classes_number",
                                2,
                                "binary classifier (Default: 2)")

        tf.flags.DEFINE_string("kmer_sizes",
                               str(input_arguments.kmer_sizes),
                               "Comma-separated filter sizes")

        # Training parameters
        tf.flags.DEFINE_float("percvalidation",
                              0.15,
                              "percentage of the training data to use for validation (Default: 15%)")

        tf.flags.DEFINE_integer("batch_size",
                                128,
                                "Batch Size (Default: 128)")

        tf.flags.DEFINE_integer("rnn_n_hidden",
                                64,
                                "hidden layers in LSTM (Default: 64)")

        tf.flags.DEFINE_float("rnn_keep_prob",
                              1.0,
                              "prob in LSTM (Default: 1.0)")

        dropout_keep_prob = 0.5
        tf.flags.DEFINE_float("dropout_keep_prob",
                              dropout_keep_prob,
                              "Dropout keep probability (Default: 0.5)")

        num_epochs = 500
        if int(input_arguments.num_epochs) > 0:
            num_epochs = int(input_arguments.num_epochs)
        tf.flags.DEFINE_integer("num_epochs",
                                num_epochs,
                                "Maximum number of training epochs (Default: 50)")

        evaluate_every = 50
        if int(input_arguments.evaluate_every) > 0:
            evaluate_every = int(input_arguments.evaluate_every)
        tf.flags.DEFINE_integer("evaluate_every",
                                evaluate_every,
                                "Evaluate model on validation set after this many steps (Default: 50)")

        save_n_models = 5
        if int(input_arguments.save_n_models) > 0:
            save_n_models = int(input_arguments.save_n_models)
        tf.flags.DEFINE_integer("num_checkpoints",
                                save_n_models,
                                "Number of checkpoints models to store (Default: 5)")

        early_stop = False
        if str(input_arguments.early_stop).strip().lower() == "true":
            early_stop = True
        tf.flags.DEFINE_boolean("early_stop",
                                early_stop,
                                "check for improvement on validation to trigger early stop (Default: False)")
        return tf.flags.FLAGS
    elif type == "CNN_LSTM":
        print("Antüshii jaya")

        input_arguments = process_param_file(paramfile)

        required_fields = ["dataset_name",
                           "input_path",
                           "kmer_sizes"]

        missing_fields = [req for req in required_fields if req not in input_arguments._fields]

        if len(missing_fields) > 0:
            print("FATAL! missing required parameters: " + missing_fields)
            sys.exit(1)
        else:
            input_k = list(map(int, input_arguments.kmer_sizes.split(",")))
            if len(input_k) > 1:
                print("FATAL! wrong kmer_sizes parameter: " + input_arguments.kmer_sizes)
                print("input one integer e.g., 9")
                sys.exit(1)


        # Data loading params
        tf.flags.DEFINE_string("model_type",
                               type,
                               "Type of model.")

        if "dataset_name" in input_arguments._fields:
            tf.flags.DEFINE_string("dataset_name",
                                   input_arguments.dataset_name,
                                   "Name for the data.")
        else:
            print("Provide a dataset_name")
            sys.exit(1)

        if "input_path" in input_arguments._fields:
            tf.flags.DEFINE_string("input_path",
                                   input_arguments.input_path,
                                   "Path for the data.")

        # Network configuration parameters
        tf.flags.DEFINE_boolean("allow_soft_placement",
                                True,
                                "Allow device soft device placement (Default: True)")
        tf.flags.DEFINE_boolean("log_device_placement",
                                False,
                                "Log placement of ops on devices (Default: False)")
        tf.flags.DEFINE_boolean("fail_train_size",
                                False,
                                "Default option, allows model training (Default: False)")

        # Model hyperparameters
        tf.flags.DEFINE_integer("feature_number",
                                4,
                                "expected input are strings with valid alphabet ATCG, " +
                                "anything else is assumed to be N (Default: 4)")

        tf.flags.DEFINE_integer("classes_number",
                                2,
                                "binary classifier (Default: 2)")

        tf.flags.DEFINE_integer("rnn_n_hidden",
                                64,
                                "hidden layers in LSTM (Default: 64)")

        tf.flags.DEFINE_float("rnn_keep_prob",
                              1.0,
                              "prob in LSTM (Default: 1.0)")

        if "kmer_sizes" in input_arguments._fields:
            tf.flags.DEFINE_string("kmer_sizes",
                                   str(input_arguments.kmer_sizes),
                                   "Comma-separated filter sizes")

        # Training parameters - don't use more than 30% of the training data for validation
        percvalidation = 0.15
        if "percvalidation" in input_arguments._fields:
            try:
                if not math.isnan(float(input_arguments.percvalidation.strip())):
                    val = round(float(input_arguments.percvalidation.strip()), 2)
                    if val > 0.0:
                        if val <= 0.30:
                            percvalidation = val
            except TypeError:
                pass
        tf.flags.DEFINE_float("percvalidation",
                              percvalidation,
                              "percentage of the training data to use for validation (Default: 15%)")

        batch_size = 128
        if "batch_size" in input_arguments._fields:
            try:
                if int(input_arguments.batch_size.strip()) > 0:
                    batch_size = int(input_arguments.batch_size.strip())
            except TypeError:
                "Can't convert provided param batch_size to integer"
        tf.flags.DEFINE_integer("batch_size",
                                batch_size,
                                "Batch Size (Default: 128)")

        dropout_keep_prob = 0.5
        tf.flags.DEFINE_float("dropout_keep_prob",
                              dropout_keep_prob,
                              "Dropout keep probability (Default: 0.5)")

        num_epochs = 50
        if "num_epochs" in input_arguments._fields:
            try:
                if int(input_arguments.num_epochs.strip()) > 0:
                    num_epochs = int(input_arguments.num_epochs.strip())
            except TypeError:
                "Can't convert provided param num_epochs to integer"
        tf.flags.DEFINE_integer("num_epochs",
                                num_epochs,
                                "Maximum number of training epochs")

        evaluate_every = 10
        if "evaluate_every" in input_arguments._fields:
            try:
                if int(input_arguments.evaluate_every) > 0:
                    evaluate_every = int(input_arguments.evaluate_every.strip())
            except TypeError:
                pass
        tf.flags.DEFINE_integer("evaluate_every",
                                evaluate_every,
                                "Evaluate model on validation set after this many steps")

        early_stop = False
        if "early_stop" in input_arguments._fields:
            try:
                if str(input_arguments.early_stop).strip().lower() == "true":
                    early_stop = True
            except TypeError:
                pass
        tf.flags.DEFINE_boolean("early_stop",
                                early_stop,
                                "check for improvement on validation to trigger early stop")

        save_n_models = 5
        if "save_n_models" in input_arguments._fields:
            try:
                if int(input_arguments.save_n_models.strip()) > 0:
                    save_n_models = int(input_arguments.save_n_models.strip())
            except TypeError:
                pass
        tf.flags.DEFINE_integer("num_checkpoints",
                                save_n_models,
                                "Save model this many time")

        # Testing parameters
        test_and_train = False
        if "test_and_train" in input_arguments._fields:
            try:
                if str(input_arguments.test_and_train).strip().lower() == "true":
                    test_and_train = True
            except TypeError:
                pass
        tf.flags.DEFINE_boolean("test_and_train",
                                test_and_train,
                                "Proceed to test after training")
        return tf.flags.FLAGS
    elif type == "RNN":
        input_arguments = process_param_file(paramfile)

        # Data loading params
        tf.flags.DEFINE_string("dataset_name",
                               input_arguments.dataset_name,
                               "Name for the data.")

        # Model Hyperparameters
        tf.flags.DEFINE_integer("feature_number",
                                4,
                                "expected input are strings with valid alphabet ATCG, " +
                                "anything else is assumed to be N (Default: 4)")

        tf.flags.DEFINE_integer("classes_number",
                                2,
                                "binary classifier (Default: 2)")

        tf.flags.DEFINE_string("kmer_sizes",
                               str(input_arguments.kmer_sizes),
                               "Comma-separated filter sizes")

        # Training parameters
        tf.flags.DEFINE_float("percvalidation",
                              0.15,
                              "percentage of the training data to use for validation (Default: 15%)")

        tf.flags.DEFINE_integer("batch_size",
                                128,
                                "Batch Size (Default: 128)")

        tf.flags.DEFINE_integer("rnn_n_hidden",
                                64,
                                "hidden layers in LSTM (Default: 64)")

        tf.flags.DEFINE_float("rnn_keep_prob",
                              1.0,
                              "prob in LSTM (Default: 1.0)")

        num_epochs = 500
        if int(input_arguments.num_epochs) > 0:
            num_epochs = int(input_arguments.num_epochs)
        tf.flags.DEFINE_integer("num_epochs",
                                num_epochs,
                                "Maximum number of training epochs (Default: 50)")

        evaluate_every = 50
        if int(input_arguments.evaluate_every) > 0:
            evaluate_every = int(input_arguments.evaluate_every)
        tf.flags.DEFINE_integer("evaluate_every",
                                evaluate_every,
                                "Evaluate model on validation set after this many steps (Default: 50)")

        save_n_models = 5
        if int(input_arguments.save_n_models) > 0:
            save_n_models = int(input_arguments.save_n_models)
        tf.flags.DEFINE_integer("num_checkpoints",
                                save_n_models,
                                "Number of checkpoints models to store (Default: 5)")

        # Misc Parameters
        tf.flags.DEFINE_boolean("allow_soft_placement",
                                True,
                                "Allow device soft device placement (Default: True)")
        tf.flags.DEFINE_boolean("log_device_placement",
                                False,
                                "Log placement of ops on devices (Default: False)")
        tf.flags.DEFINE_boolean("fail_train_size",
                                False,
                                "Default allows to train a model (Default: False)")

        tf.flags.DEFINE_boolean("test_and_train",
                                False,
                                "Proceed to test after training (Default: False)")

        early_stop = False
        if str(input_arguments.early_stop).strip().lower() == "true":
            early_stop = True
        tf.flags.DEFINE_boolean("early_stop",
                                early_stop,
                                "check for improvement on validation to trigger early stop (Default: False)")

        return tf.flags.FLAGS


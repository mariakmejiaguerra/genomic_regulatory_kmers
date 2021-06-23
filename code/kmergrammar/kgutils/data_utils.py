# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' # Introduction
# ' Some of the code below is still under active development

# ' ## Required libraries
# + name = 'import_libraries', echo=False
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sqlalchemy
import sqlite3
import logging

from itertools import product
from math import log
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# TODO: add visualization to LSTMs and CNNs outputs as well as for W2V and bagOfKmers
# check here: https://github.com/HendrikStrobelt/LSTMVis

def batch_index(seq, size):
    batchs = [seq[pos:pos + size] for pos in range(0, len(seq), size)]
    return [[i[0], i[-1]] for i in batchs]


def onehotencoded_record(string, matrix, record):
    """
    write ATCG and N values as a column in a matrix 4 by len(string)
    :param string: DNA string with ATCG or N (or any other non-ATCG character)
    :param matrix: numpy matrix to be filled
    :param record: integer
    :return numpy.ndarray
    """
    alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    position = 0
    for letter in string:
        if letter in alphabet:
            matrix[record, position + alphabet.get(letter)] = 1.0
        else:  # i.e., 'N'
            matrix[record, position + 0] = 0.25
            matrix[record, position + 1] = 0.25
            matrix[record, position + 2] = 0.25
            matrix[record, position + 3] = 0.25
        position = position + 4
    return matrix


def label_record(label, matrix, record):
    """
    write 1 and 0 for the two classes in the
    :param label: class is 1 or 0
    :param matrix: numpy matrix to be filled
    :param record: integer
    :return uniq_kmers: list of sorted unique kmers
    """
    if label == 1:
        matrix[record, 0] = 0.0
        matrix[record, 1] = 1.0
    else:  # everything else is classed as not-bound
        matrix[record, 0] = 1.0
        matrix[record, 1] = 0.0
    return matrix


def createKmerSet(kmersize):
    """
    write all possible kmers
    :param kmersize: integer, 8
    :return uniq_kmers: list of sorted unique kmers
    """
    kmerSet = set()
    nucleotides = ["a", "c", "g", "t"]
    kmerall = product(nucleotides, repeat=kmersize)
    for i in kmerall:
        kmer = ''.join(i)
        kmerSet.add(kmer)
    uniq_kmers = sorted(list(kmerSet))
    return uniq_kmers


def compute_kmer_entropy(kmer):
    """
    compute shannon entropy for each kmer
    :param kmer: string
    :return entropy: float
    """
    prob = [float(kmer.count(c)) / len(kmer) for c in dict.fromkeys(list(kmer))]
    entropy = - sum([p * log(p) / log(2.0) for p in prob])
    return round(entropy, 2)


# entropy filter should be calibrated
def make_stopwords(kmersize):
    """
    write filtered out kmers
    :param kmersize: integer, 8
    :return stopwords: list of sorted low-complexity kmers
    """
    kmersize_filter = {5: 1.3, 6: 1.3, 7: 1.3, 8: 1.3, 9: 1.3, 10: 1.3}
    limit_entropy = kmersize_filter.get(kmersize)
    kmerSet = set()
    nucleotides = ["a", "c", "g", "t"]
    kmerall = product(nucleotides, repeat=kmersize)
    for n in kmerall:
        kmer = ''.join(n)

        if compute_kmer_entropy(kmer) < limit_entropy:
            kmerSet.add(make_newtoken(kmer))
        else:
            continue
    stopwords = sorted(list(kmerSet))
    return stopwords


def createNewtokenSet(kmersize):
    """
    write all possible newtokens
    :param kmersize: integer, 8
    :return uniq_newtokens: list of sorted unique newtokens
    """
    newtokenSet = set()
    uniq_kmers = createKmerSet(kmersize)
    for kmer in uniq_kmers:
        newtoken = make_newtoken(kmer)
        newtokenSet.add(newtoken)
    uniq_newtokens = sorted(list(newtokenSet))
    return uniq_newtokens


def make_newtoken(kmer):
    """
    write a collapsed kmer and kmer reverse complementary as a newtoken
    :param kmer: string e.g., "AT"
    :return newtoken: string e.g., "atnta"
    :param kmer: string e.g., "TA"
    :return newtoken: string e.g., "atnta"
    """
    kmer = str(kmer).lower()
    newtoken = "n".join(sorted([kmer, kmer.translate(str.maketrans('tagc', 'atcg'))[::-1]]))
    return newtoken


def write_ngrams(sequence, glob_kmerlength):
    """
    write a bag of newtokens of size n
    :param sequence: string e.g., "ATCG"
    :param (intern) kmerlength e.g., 2
    :return newtoken_string: string e.g., "atnta" "gatc" "cgcg"
    """
    seq = str(sequence).lower()
    finalstart = (len(seq) - glob_kmerlength) + 1
    allkmers = [seq[start:(start + glob_kmerlength)] for start in range(0, finalstart)]
    tokens = [make_newtoken(kmer) for kmer in allkmers if len(kmer) == glob_kmerlength and "n" not in kmer]
    newtoken_string = " ".join(tokens)
    return newtoken_string


def tokenize_data(train_file, test_file, kmerlength, logger):
    dftrain = pd.read_csv(train_file,
                          sep='\t',
                          header=0,
                          names=["chr_num", "left_idx", "right_idx", "dna_string", "bound"])
    print("training set is ready")
    # validation or test

    dftest = pd.read_csv(test_file,
                         sep='\t',
                         header=0,
                         names=["chr_num", "left_idx", "right_idx", "dna_string", "bound"])
    print("test set is ready")

    print("Collecting tokens")
    dftrain["tokens"] = np.vectorize(write_ngrams)(dftrain["dna_string"], kmerlength)
    # it's also possilbe to write dftrain.apply(lambda x: write_ngrams(x['dna_string'], glob_kmerlength), axis=1)
    dftest["tokens"] = np.vectorize(write_ngrams)(dftest["dna_string"], kmerlength)
    tokenized_train_points = dftrain["tokens"].tolist()
    tokenized_test_points = dftest["tokens"].tolist()

    print("Collecting labels")
    train_labels = dftrain["bound"].tolist()
    test_labels = dftest["bound"].tolist()
    unique_train_labels = len(list(set(train_labels)))
    unique_test_labels = len(list(set(test_labels)))

    # Check that labels are as many as expected for binary classification
    if unique_train_labels < 2 or unique_test_labels < 2:
        print("ERROR: Expected 2 train and test labels. Got %d train labels and %d test labels" % (
        unique_train_labels, unique_test_labels))
        logger.info("Unique train labels = %d" % unique_train_labels)
        logger.info("Unique test labels = %d" % unique_test_labels)
        quit()
    else:
        return tokenized_train_points, train_labels, tokenized_test_points, test_labels


def get_vocabulary(kmerlength, full, logger):
    print("Building a vocabulary from tokens")
    all_tokens = createNewtokenSet(kmerlength)
    tmpvectorizer = TfidfVectorizer(min_df=1, max_df=1.0, sublinear_tf=True, use_idf=True)
    tmpvectorizer.fit_transform(all_tokens)  # newtoken sequences to numeric index.
    vcblry = tmpvectorizer.get_feature_names()

    # Remove stopwords if full is false
    if full:
        print("keeping all low-complexity k-mers")
        kmer_names = vcblry
        feature_names = np.asarray(kmer_names)  # key transformation to use the fancy index into the report
    else:
        stpwrds = make_stopwords(kmerlength)
        print("removing %d low-complexity k-mers" % len(stpwrds))
        kmer_names = [x for x in vcblry if x not in stpwrds]
        feature_names = np.asarray(kmer_names)  # key transformation to use the fancy index into the report

    # Check that tokens are as many as expected
    expected_tokens = math.pow(4, kmerlength) / 2
    if len(kmer_names) > expected_tokens:
        print("ERROR: Expected %d tokens. Obtained %d tokens" % (expected_tokens, len(kmer_names)))
        logger.info("Expecting %d tokens" % expected_tokens)
        logger.info("Feature index contains %d tokens" % len(kmer_names))
        logger.info("ERROR: expected %d tokens, got %d tokens" % (expected_tokens, len(kmer_names)))
        logger.info("ERROR: More features than expected!")
        quit()
    else:
        print("Expected %d tokens. Obtained %d tokens" % (expected_tokens, len(kmer_names)))
        logger.info("Feature index contains %d tokens" % len(kmer_names))

    return kmer_names, feature_names


def TFIDF_data(tokenized_train_points, train_labels, tokenized_test_points, test_labels, full, kmerlength, logger):
    Y_DEV = np.asarray(train_labels)
    Y_holdout = np.asarray(test_labels)
    token_names, tokens = get_vocabulary(kmerlength, full)

    print("Extracting features from the training data using TfidfVectorizer")
    vectorizer = TfidfVectorizer(min_df=1, max_df=1.0, sublinear_tf=True, use_idf=True,
                                 vocabulary=token_names)  # vectorizer for kmer frequencies
    X_TFIDF_DEV = vectorizer.fit_transform(tokenized_train_points)
    print("train_samples: %d, n_features: %d" % X_TFIDF_DEV.shape)
    print("Positive n_labels: %d Negative n_labels: %d" % (train_labels.count(0), train_labels.count(1)))

    print("Extracting features from the holdout data using TfidfVectorizer")
    X_TFIDF_test = vectorizer.fit_transform(tokenized_test_points)
    print("test_samples: %d, n_features: %d" % X_TFIDF_test.shape)
    print("Positive n_labels: %d Negative n_labels: %d" % (train_labels.count(0), train_labels.count(1)))

    logger.info("Train dataset")
    logger.info("n_samples: %d, n_features: %d" % X_TFIDF_DEV.shape)
    logger.info("Positive n_labels: %d Negative n_labels: %d" % (test_labels.count(0), test_labels.count(1)))
    logger.info("Test dataset")
    logger.info("n_samples: %d, n_features: %d" % X_TFIDF_test.shape)
    logger.info("Positive n_labels: %d Negative n_labels: %d" % (test_labels.count(0), test_labels.count(1)))

    return X_TFIDF_DEV, Y_DEV, X_TFIDF_test, Y_holdout, tokens


# Plotting
# ==================================================

def save_plot_prc(precision, recall, avg_prec, figure_file, name):
    '''
    plot precission recall curve
    :param precission: precission
    :param recall: recall
    :param avg_prec: avg_prec
    :param figure_file: figure_file
    :param name: name
    '''
    # plt.close('all')
    plt.clf()
    title = 'Precision Recall Curve - double strand ' + name
    plt.title(title)
    plt.plot(recall, precision, label='Precission = %0.2f' % avg_prec)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(figure_file)


def save_validation_plot_roc(fprs, tprs, aucs, figure_file, title):
    '''
    plot roc curve for validation steps
    :param fprs: list of arrays with false positive rate across validation
    :param tprs: list of arrays with true positive rate across validation
    :param aucs: list of auc values across validation
    :param figure_file: figure_file
    '''
    print("plotting validation across training")
    ax = plt.subplot(111)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.suptitle(title, y=1.05, fontsize=14)
    plt.title('Receiver Operating Characteristic - Validation', fontsize=10)
    for idx, fpr in enumerate(fprs):
        tpr = tprs[idx]
        ax.plot(fpr, tpr, lw=1, alpha=0.3)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Random', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.legend(loc='lower right')
    plt.savefig(figure_file)


def save_plot_roc(fpr, tpr, auc, figure_file, title):
    '''
    plot roc curve for final test
    :param fpr: array with false positive rate across testing
    :param tpr: array with true positive rate across validation
    :param auc: auc across testing
    :param figure_file: figure_file
    '''
    print("plotting test")
    plt.suptitle(title, y=1.05, fontsize=14)
    plt.title('Receiver Operating Characteristic - Test', fontsize=10)
    plt.plot(fpr, tpr, lw=3, alpha=0.6, color='k', label='ROC test = %0.2f' % (auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(figure_file)
    plt.close("all")


# Load data
# ==================================================
def data_in_dir(input_path, base_data):
    """
    after running preprocessing it is possible to have more than one database
    database_in_folder
    """
    data = []
    base_dir = ''.join([input_path, base_data, '/'])
    for file in os.listdir(base_dir):
        if file.endswith(".db"):
            input_path, dataset = os.path.split(file)
            datum_idx = dataset.replace(".db", "")
            data.append(datum_idx)
    return base_dir, data


# + name = 'get_features_and_labels', echo=False
def get_features_and_labels(data_name, data_files, logger, train=True):
    """Load train and test data

    Parameters
    ----------

    Returns
    -------

    """
    if not os.path.exists(data_files.get("database")):
        dbfile = data_files.get("database")
        logger.info("DATABASE FILE {} DOESN'T EXIST".format(os.path.abspath(dbfile)))
        print("Fatal: DATABASE FILE {} DOESN'T EXIST, CAN'T TRAIN A MODEL".format(os.path.abspath(dbfile)))
        sys.exit(1)
    else:
        dbfile = data_files.get("database")
        logger.info("READING DATABASE FILE {}".format(os.path.abspath(dbfile)))

    with sqlite3.connect(dbfile) as db:
        df_tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", db)
    if train:
        if 'validation' in df_tables['name'].unique():
            query = "SELECT * FROM train UNION ALL SELECT * FROM validation"
            logger.info("reading train and validation table")
        else:
            query = "SELECT * FROM train"
            logger.info("reading train table")
    else:
        query = "SELECT * FROM test"
        logger.info("reading test table")

    with sqlite3.connect(dbfile) as db:
        dfdata = pd.read_sql_query(query, db)
        df = dfdata[["dna_string", "bound"]]
        df = shuffle(df)
        sequence_length = df["dna_string"].map(lambda x: len(x)).max()
        logger.info("dna_string of length {}".format(sequence_length))
        num_lines = df.shape
        logger.info("rows in dataframe {}".format(num_lines[0]))
        num_bases = sequence_length * 4

        # preparing numpy array for the image of shape num_sequences X num_bases and type 'float32'
        matrix_features = np.empty([num_lines[0], num_bases], dtype=np.dtype('float32'))
        # initialize entries to zero
        matrix_features.fill(0.0)
        # preparing numpy array for labels of shape num_sequences X 2 and type 'float32' without initializing entries
        matrix_labels = np.empty([num_lines[0], 2], dtype=np.dtype('float32'))

        for num_record, record in enumerate(df.to_dict("records")):
            dna_string = str(record["dna_string"])
            bound = int(record["bound"])
            matrix_features = onehotencoded_record(dna_string, matrix_features, num_record)
            matrix_labels = label_record(bound, matrix_labels, num_record)

    return matrix_features, matrix_labels, sequence_length


# Save results and models
# ==================================================
def save_weighted_kmers_to_db(TFIDF_LR, feature_names, dbfile_name):
    print("Export the kmer weights from the LR classifier to a sqlite3 database")
    outengine = 'sqlite:///' + dbfile_name
    disk_engine = sqlalchemy.create_engine(outengine)
    if hasattr(TFIDF_LR, 'coef_'):
        top = np.argsort(TFIDF_LR.coef_[0])[-5:]  # select the top 5 index
        botton = np.argsort(TFIDF_LR.coef_[0])[:5]  # select the bottom 5 index
        logging.info("database table LR_results")
        logging.info("top 5 positive kmers")
        logging.info(" ".join([i.split('n')[0].upper() for i in feature_names[top]]))
        logging.info(" ".join([i.split('n')[1].upper() for i in feature_names[top]]))
        logging.info("top 5 negative kmers")
        logging.info(" ".join([i.split('n')[0].upper() for i in feature_names[botton]]))
        logging.info(" ".join([i.split('n')[1].upper() for i in feature_names[botton]]))
        print("Saving data to database table LR_results")
        print('*' * 80)
        print("%s: %s" % ("pos kmers", " ".join([i.split('n')[0].upper() for i in feature_names[top]])))
        print("%s: %s" % ("pos kmers", " ".join([i.split('n')[1].upper() for i in feature_names[top]])))
        print()  # making room
        print("%s: %s" % ("neg kmers", " ".join([i.split('n')[0] for i in feature_names[botton]])))
        print("%s: %s" % ("neg kmers", " ".join([i.split('n')[1] for i in feature_names[botton]])))
        print('*' * 80)
        print()  # making room
        LR_weights = []
        for idx, kmer_score in enumerate(TFIDF_LR.coef_[0]):
            features = feature_names[idx].split('n')
            LR_weights.append({'kmer': features[0].upper(), 'revcomp': features[1].upper(), 'score': kmer_score})
        LR_weights_feature = pd.DataFrame(LR_weights)
        LR_weights_feature.to_sql('LR_results', disk_engine, if_exists='append')
        df = pd.read_sql_query('SELECT * FROM LR_results LIMIT 3', disk_engine)
        print(df.head())


def save_kmers_weigths(TFIDF_LR, LR_model_file):
    print("Save the LR classifier to a pkl file to be used in another Python process")
    joblib.dump(TFIDF_LR, LR_model_file)
    logging.info("Saving model to disk")
    logging.info(LR_model_file)
    # Later load back the pickled model (possibly in another Python process) with:
    # clf = joblib.load('filename.pkl')


# + name= 'evaluate_prediction', echo=False
def evaluate_prediction(true_labels, pred_labels, prev_auc=0.0001):
    accuracy = metrics.accuracy_score(np.argmax(true_labels, axis=1),
                                      np.argmax(pred_labels, axis=1))

    fpr, tpr, thresholds = metrics.roc_curve(true_labels[:, 0],
                                             pred_labels[:, 0],
                                             pos_label=1.0)
    roc_auc = metrics.auc(fpr, tpr)

    avg_precision = metrics.average_precision_score(true_labels[:, 0],
                                                    pred_labels[:, 0])
    perc_chg_auc = (roc_auc - prev_auc) / prev_auc

    evaluate_dict = {"fpr": fpr,
                     "tpr": tpr,
                     "accuracy": accuracy,
                     "roc_auc": roc_auc,
                     "avg_prec": avg_precision,
                     "perc_chg_auc": perc_chg_auc}

    return evaluate_dict
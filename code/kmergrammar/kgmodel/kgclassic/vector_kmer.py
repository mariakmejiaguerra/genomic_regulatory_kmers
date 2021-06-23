# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' # Introduction
# ' Some of the code below is still under active development

# ' ## Required libraries
# + name = 'import_libraries', echo=False

import os
import sys
import glob

import numpy as np
import pandas as pd
import sqlalchemy
import multiprocessing
import gensim
import time

from copy import deepcopy
from itertools import product
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold

from kmergrammar.kgutils import helpers


# + name= 'hello_world', echo=False
def hello_world():
    print("Buorist båhtem")

# + name= 'get_k_size', echo=False
def get_k_size():
    """
    Returns
    -------
    KMERSIZE: `int`
        global variable
    """
    return KMERSIZE


# + name = 'make_newtoken', echo=False
def make_newtoken(kmer):
    """make_newtoken
    
    Parameters
    ----------
    kmer: `string`
        
    Returns
    -------
    newtoken: `string`
          
    """
    s = set("ACGT")  # if odd characters return token
    kmer = str(kmer).upper()
    if not set(kmer).difference(s):
        newtoken = "n".join(sorted([kmer, kmer.translate(str.maketrans("TAGC", "ATCG"))[::-1]]))
        return newtoken
    else:
        return "token"


# + name = 'split_len', echo=False
def split_len(seq):
    """To slide through a sequence and extract k-mers of length KMERSIZE
    
    Parameters
    ----------
    seq: `str`
    
    Returns
    -------
    `list str`
    """
    length = KMERSIZE
    return [''.join(x) for x in zip(*[list(seq[z::length]) for z in range(length)])]


# + name = 'merge_two_dicts', echo=False
def merge_two_dicts(x, y):
    """Given two dicts x and y, merge them into a new dict z, as a shallow copy
    
    Parameters
    ----------
    x: `dict()`
    y: `dict()`
    
    Returns
    -------
    z: `dict()`
    """
    z = x.copy()
    z.update(y)

    return z


# + name = 'write_sentences', echo=False
def write_sentences(sequence):
    """Write_sentences with newtokens
    
    Parameters
    ----------
    
    Returns
    -------
    """
    kmersize = get_k_size()
    seq = str(sequence).lower()
    sentences = []
    if len(seq) > 0:
        first_sentence_kmers = split_len(seq)
        alltokens = [make_newtoken(kmer) for kmer in first_sentence_kmers if len(kmer) == kmersize]
        first_sentence_newtokens = [newtoken for newtoken in alltokens if newtoken != "token"]
        sentences.append(first_sentence_newtokens)  # each sentence is a list
        n = kmersize - 1
        while n >= 1:
            next_sentence_kmers = split_len(seq[n:])
            alltokens = [make_newtoken(kmer) for kmer in next_sentence_kmers if len(kmer) == kmersize]
            next_sentence_newtokens = [newtoken for newtoken in alltokens if newtoken != "token"]
            sentences.append(next_sentence_newtokens)
            n = n - 1

    return sentences


# + name = 'write_kmer_sentences', echo=False
def write_kmer_sentences(sequence):
    """Write_sentences with kmers
    
    Parameters
    ----------
    
    Returns
    -------    
    """
    kmersize = get_k_size()
    seq = str(sequence).lower()
    sentences = []
    if len(seq) > 0:
        first_sentence_kmers = split_len(seq)
        alltokens = [make_newtoken(kmer) for kmer in first_sentence_kmers if len(kmer) == kmersize]
        first_sentence_newtokens = [newtoken for newtoken in alltokens if newtoken != "token"]
        sentences.append(first_sentence_newtokens)  # each sentence is a list
        n = kmersize - 1
        while n >= 1:
            next_sentence_kmers = split_len(seq[n:])
            alltokens = [make_newtoken(kmer) for kmer in next_sentence_kmers if len(kmer) == kmersize]
            next_sentence_newtokens = [newtoken for newtoken in alltokens if newtoken != "token"]
            sentences.append(next_sentence_newtokens)
            n = n - 1

    return sentences


# + name = 'createKmerSet', echo=False
def createKmerSet(kmersize):
    """Write all possible kmers
    
    Parameters
    ----------
    
    Returns
    -------    
    """
    kmerSet = set()
    nucleotides = ["A", "C", "G", "T"]
    kmerall = product(nucleotides, repeat=kmersize)
    for i in kmerall:
        kmer = ''.join(i)
        kmerSet.add(kmer)
    uniq_kmers = sorted(list(kmerSet))

    return uniq_kmers


# + name = 'createNewtokenSet', echo=False
def createNewtokenSet(kmersize):
    """Write all possible newtokens
    
    Parameters
    ----------
    
    Returns
    -------        
    """
    newtokenSet = set()
    uniq_kmers = createKmerSet(kmersize)
    for kmer in uniq_kmers:
        newtoken = make_newtoken(kmer)
        newtokenSet.add(newtoken)
    uniq_newtokens = sorted(list(newtokenSet))

    return uniq_newtokens


# + name = 'sentences_in_category', echo=False
def sentences_in_category(list_dictionaries, category):
    """Given a dictionary per document (sequence) return a generator that yield the rated sentences
    
    Parameters
    ----------
    
    Returns
    -------      
    """
    for dictionary in list_dictionaries:
        if dictionary["bound"] in category:
            for sentence in dictionary["sentences"]:
                yield sentence


# + name = 'sentences_for_vocab', echo=False
def sentences_for_vocab(uniq_tokens):
    """Given a dictionary per document (sequence) return a generator that yield the rated sentences
        
    Parameters
    ----------
    
    Returns
    -------
    """
    for token in uniq_tokens:
        sentence = [token, token, token]
        yield sentence


# + name = 'get_categories', echo=False
def get_categories(dbfile, logger, kmer_parsing=False):
    """Check that categories correspond to a binary classifier
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    # prepare input engine connection
    dbpath = "sqlite:///" + dbfile
    logger.info(dbpath)
    engine = sqlalchemy.create_engine(dbpath)
    category_query = "SELECT DISTINCT(bound) FROM train"

    with engine.connect() as conn, conn.begin():
        dfcategory = pd.read_sql_query(category_query, conn)

    dfcategory.columns = ["category"]
    categories = [int(cat) for cat in dfcategory["category"].tolist()]

    logger.info("Categories:")
    logger.info(set(categories))

    # vector-k-mer is a binary classifier, to train it two categories are expected
    if len(categories) != 2:
        logger.info("Unexpected number of categories to train a binary classifier")
        sys.exit(1)
    else:
        return categories


# + name = 'load_embeddings', echo=False
def load_embeddings(model_path, model_file_base):
    """Load pre-trained vectors
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    if not os.path.exists(model_path):
        return None
    else:
        model = []
        model_path = os.path.dirname(model_path)
        print("Loading embeddings save on:", model_path)
        for filename in glob.iglob(os.path.join(model_path, model_file_base + "*w2v")):
            model.append(gensim.models.Word2Vec.load(filename))

    if model:
        return model
    else:
        return None


# + name = 'load_data', echo=False
def load_data(dbfile, logger, kmer_parsing=False, data=1):
    """Load train and test data
    
    Parameters
    ----------
    
    Returns
    -------    
    
    """
    # expected tables in database
    table_dict = {1: "train", 2: "test"}

    # prepare input engine connection
    dbpath = "sqlite:///" + dbfile
    logger.info(dbpath)
    engine = sqlalchemy.create_engine(dbpath)

    if data in table_dict.keys():
        table = table_dict.get(data)
        with engine.connect() as conn, conn.begin():
            dfdata = pd.read_sql_table(table, conn)
    else:
        logger.info("Table No {} is not a valid input".format(data))
        print("Table No {} is not a valid input, try 1 for train, or 2 for test".format(data))
        sys.quit(1)

    dfdata.columns = ["chrom", "start", "end", "dna_string", "score", "strand", "bound", "run_id"]

    if kmer_parsing:
        dfdata["sentences"] = dfdata["dna_string"].apply(write_kmer_sentences).astype("object")
    else:
        dfdata["sentences"] = dfdata["dna_string"].apply(write_sentences).astype("object")

    logger.info("{} dataset".format(table))
    logger.info(dfdata.shape)

    return dfdata


# + name = 'set_embeddings', echo=False
def set_embeddings(kmerlength, context, logger, kmer_parsing=False, num_features=300, min_word_count=0, hs=3,
                   iterations=30, negative=0):
    """Preparing the embedding models
    
    Parameters
    ----------
    kmerlength: `int`
    context: `int`
    logger: `logging`
    kmer_parsing: `boolean`, optional
    num_features: `int`, optional
    min_word_count: `int`, optional
    hs: `int`, optional
    iterations: `int`, optional
    negative: `int`, optional
    
    Returns
    ------- 
    basemodel: `gensim.models.Word2Vec`
    
    """
    assert gensim.models.word2vec.FAST_VERSION > -1  # This will be painfully slow otherwise
    num_workers = multiprocessing.cpu_count()  # go and use all the cores

    # base word2vec model
    print("Preparing the base model ...")
    basemodel = gensim.models.Word2Vec(workers=num_workers, iter=iterations, hs=hs, negative=negative,
                                       size=num_features, min_count=min_word_count, window=context)

    # kmer or collapsed k-mers and rev-complementary k-mers as tokens
    if kmer_parsing:
        all_tokens = createKmerSet(kmerlength)
        expected_newtokens = len(all_tokens)
        logger.info("input: kmerlength - {}".format(kmerlength))
    else:
        all_tokens = createNewtokenSet(kmerlength)
        expected_newtokens = len(all_tokens)
        logger.info("input: newtoken size - {}".format(str(int(kmerlength * 2))))

    # building model vocabulary
    logger.info("Building vocabulary")
    basemodel.build_vocab(sentences_for_vocab(all_tokens))
    logger.info(basemodel)

    if len(basemodel.wv.vocab) > expected_newtokens:
        print("ERROR: Expected {} tokens. Obtained {} tokens".format(expected_newtokens, len(basemodel.wv.vocab)))
        logger.info("ERROR: expected {} tokens, got {} tokens".format(expected_newtokens, len(basemodel.wv.vocab)))
        logger.info("ERROR: More features than expected!")
        sys.exit(1)
    else:
        print(basemodel)
        print("Vocabulary: Expected {} tokens. Obtained {} tokens".format(expected_newtokens, len(basemodel.wv.vocab)))
        logger.info("Vocabulary: expected {} tokens, got {} tokens".format(expected_newtokens, len(basemodel.wv.vocab)))

    return basemodel


# TODO: parallelize me! Needs to be improved, so far everything is done in-memory!
# + name = 'docprob', echo=False
def docprob(seqtest, models):
    """Function to test the vector-k-mer model, it uses the w2v output (models) and bayes inversion
    
    Parameters
    ----------
    seqtest: `pd.DataFrame`
        A list of regions, and each region is a list of sentences
    models: `list` 
        Each list is a gensim.models.Word2Vec (each potential class)
    
    Returns
    -------
    prob: `pd.DataFrame`
    A dataframe with five columns
    "category_0": `float` The probability for the first model
    "category_1": `float The probability for the second model
    "true_category": `int`, 0 for the first model or 1 for the second model
    "predict": `int`, 0 for the first model or 1 for the second model
    "predict_proba": `float The probability for the winner model
    """
    docs = [r["sentences"] for r in seqtest]
    docs_cats = pd.Series([r["bound"] for r in seqtest])
    sentlist = [s for d in docs for s in d]
    llhd = np.array([m.score(sentlist, len(sentlist)) for m in models])
    lhd = np.exp(llhd - llhd.max(axis=0))
    prob = pd.DataFrame((lhd / lhd.sum(axis=0)).transpose())
    prob["seq"] = [i for i, d in enumerate(docs) for s in d]
    prob = prob.groupby("seq").mean()
    prob["true_category"] = docs_cats.values
    prob["predict"] = np.where(prob[1] <= prob[0], 0, 1)
    prob["predict_proba"] = np.where(prob[1] <= prob[0], prob[0], prob[1])
    prob.columns = ["category_0", "category_1", "true_category", "predict", "predict_proba"]

    return prob


# + name = 'test_vectors', echo=False
def test_vectors(test_set, catmodels, logger):
    """Classifier 
    
    Parameters
    ----------
    seqtest: `pd.DataFrame`
        A list of regions, and each region is a list of sentences
    models: `list` 
        Each list is a gensim.models.Word2Vec (each potential class)
    
    Returns
    -------
    prob: `pd.DataFrame`
    A dataframe with five columns
    "category_0": `float` The probability for the first model
    "category_1": `float The probability for the second model
    "true_category": `int`, 0 for the first model or 1 for the second model
    "predict": `int`, 0 for the first model or 1 for the second model
    "predict_proba": `float The probability for the winner model
    """
    # predict with word2vec and bayesinversion.
    np.random.shuffle(test_set)
    print("Predicted labels for holdout set")
    w2v_hold_df = docprob(test_set, catmodels)
    w2v_hold_true = w2v_hold_df["true_category"].tolist()
    w2v_hold_pred = w2v_hold_df["predict"].tolist()
    w2v_hold_prob = w2v_hold_df["category_1"].tolist()

    print("Model Evaluation:")
    print(metrics.classification_report(w2v_hold_true, w2v_hold_pred))
    print("Accuracy score")
    print(metrics.accuracy_score(w2v_hold_true, w2v_hold_pred))
    print("ROC_AUC")
    print(metrics.roc_auc_score(w2v_hold_true, w2v_hold_prob))

    logger.info("Evaluation report")
    logger.info(metrics.classification_report(w2v_hold_true, w2v_hold_pred))
    logger.info("ROC_AUC")
    logger.info(metrics.roc_auc_score(w2v_hold_true, w2v_hold_prob))
    logger.info("Accuracy score")
    logger.info(metrics.accuracy_score(w2v_hold_true, w2v_hold_pred))

    return w2v_hold_true, w2v_hold_pred, w2v_hold_prob


# + name = 'train_vectors', echo=False
def train_vectors(training_set, categories, model_file_base, args, logger):
    """Train vectors
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    basemodel = set_embeddings(int(args.ksize), int(args.wsize), logger, kmer_parsing=args.kmer_parsing)
    catmodels = [deepcopy(basemodel) for each in categories]

    print("Iterating through categories to train the vectors ...")

    t0 = time.time()
    for category in categories:
        logger.info("building/saving model for category {} :".format(str(category)))
        slist = list(sentences_in_category(training_set, [category]))
        logger.info("category with {} examples".format(len(slist)))
        catmodels[category].train(slist, total_examples=len(slist), epochs=basemodel.iter)
    duration = time.time() - t0
    logger.info("Training vectors done in {}s".format(round(duration, 6)))

    return catmodels


# + name = 'test_model', echo=False
def test_model(args, logger):
    """Evaluate a model
    
    Parameters
    ----------
    
    Returns
    -------    
    
    """
    global KMERSIZE
    KMERSIZE = int(args.ksize)
    kmerlength = KMERSIZE
    workingdir = args.datadir
    windowsize = int(args.wsize)
    dataset_name = args.dataset
    kmer_parsing = args.kmer_parsing

    if kmer_parsing:
        vector_type = "kmer"
    else:
        vector_type = "token"

    dbfile = os.path.join(workingdir, dataset_name + '.db')
    dftest_set = load_data(dbfile, logger, kmer_parsing=kmer_parsing, data=2)
    dftest_set = dftest_set[["sentences", "bound"]]
    test_set = dftest_set.to_dict('records')

    model_file_base = "kgrammar_vector-k-mer_model_{}_k_{}_w_{}_vtype_{}".format(dataset_name, kmerlength, windowsize,
                                                                                 vector_type)
    roc_figure_file = os.path.join(workingdir, model_file_base + "_roc.png")
    prc_figure_file = os.path.join(workingdir, model_file_base + "_prc.png")

    try:
        vectors = load_embeddings(workingdir, model_file_base)
    except:
        print("No vectors available under provided file name")
        sys.exit(1)

    w2v_hold_true, w2v_hold_pred, w2v_hold_prob = test_vectors(test_set, vectors, logger)
    fpr, tpr, _ = roc_curve(w2v_hold_true, w2v_hold_prob, pos_label=1)
    precision, recall, _ = metrics.precision_recall_curve(w2v_hold_true, w2v_hold_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    avg_prc = metrics.average_precision_score(w2v_hold_true, w2v_hold_prob)
    model_result = [{"false_positive_rate": fpr,
                     "true_positive_rate": tpr,
                     "roc_auc": roc_auc,
                     "precision": precision,
                     "recall": recall,
                     "avg_prec": avg_prc}]
    helpers.save_plot_prc(model_result, prc_figure_file, dataset_name)
    helpers.save_plot_roc(model_result, roc_figure_file, dataset_name)


# + name = 'train_model', echo=False
def train_model(args, logger):
    """Train the model and evalute with cross validation
    
    Parameters
    ----------
    
    Returns
    -------    
    results_cv
    """
    global KMERSIZE
    KMERSIZE = int(args.ksize)
    kmerlength = KMERSIZE
    workingdir = args.datadir
    windowsize = int(args.wsize)
    dataset_name = args.dataset
    kmer_parsing = bool(args.kmer_parsing)
    cv = int(args.cv)

    logger.info("input: kmerlength {}".format(kmerlength))
    logger.info("input: windowsize {}".format(windowsize))
    logger.info("input: kmer_parsing {}".format(kmer_parsing))
    logger.info("input: cross-validation {}".format(cv))

    if kmer_parsing:
        vector_type = "kmer"
    else:
        vector_type = "token"

    dbfile = os.path.join(workingdir, dataset_name + '.db')
    file_name = "kgrammar_vector-k-mer_model_{}_k_{}_w_{}_vtype_{}".format(dataset_name, kmerlength, windowsize,
                                                                           vector_type)
    roc_figure_file = os.path.join(workingdir, file_name + "cv_{}_roc.png".format(cv))
    prc_figure_file = os.path.join(workingdir, file_name + "cv_{}_prc.png".format(cv))

    if not os.path.exists(dbfile):
        logger.info("A database file to start the training is not available")
        sys.exit(1)

    logger.info("input: working directory - {}".format(workingdir))
    logger.info("input: dataset - {}".format(dataset_name))
    logger.info("input: database - {}".format(dbfile))
    categories = get_categories(dbfile, logger, kmer_parsing=kmer_parsing)
    dftraining_set = load_data(dbfile, logger, kmer_parsing=kmer_parsing, data=1)

    print("input: kmerlength {}".format(get_k_size()))
    results_cv = []

    if cv == 1:
        # Model is tested in holdout data
        dftest_set = load_data(dbfile, logger, kmer_parsing=kmer_parsing, data=2)

        dftraining_set = dftraining_set[["sentences", "bound"]]
        dftest_set = dftest_set[["sentences", "bound"]]

        test_set = dftest_set.to_dict("records")
        training_set = dftraining_set.to_dict("records")

        vector_models = train_vectors(training_set, categories, file_name, args, logger)

        for vector_class in vector_models:
            model_file = file_name + "_cat_{}".format(str(vector_class)) + ".w2v"
            vector_models[vector_class].save(os.path.join(workingdir, model_file))
            logger.info(model_file)

        w2v_hold_true, w2v_hold_pred, w2v_hold_prob = test_vectors(test_set, vector_models, logger)

        fpr, tpr, thresholds = roc_curve(w2v_hold_true, w2v_hold_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        fpr, tpr, thresholds = roc_curve(w2v_hold_true, w2v_hold_prob, pos_label=1)
        precision, recall, thresholds = metrics.precision_recall_curve(w2v_hold_true, w2v_hold_prob, pos_label=1)
        avg_prc = metrics.average_precision_score(w2v_hold_true, w2v_hold_prob)

        results_cv.append({"false_positive_rate": fpr,
                           "true_positive_rate": tpr,
                           "roc_auc": roc_auc,
                           "precision": precision,
                           "recall": recall,
                           "avg_prec": avg_prc})

        print("Done with training and testing in holdout data")
        helpers.save_plot_prc(results_cv, prc_figure_file, dataset_name)
        helpers.save_plot_roc(results_cv, roc_figure_file, dataset_name)
    elif cv > 1:
        # When CV is > 1, only the train table is queries and the holdout data (if any) ignored
        dfdata_set = load_data(dbfile, logger, kmer_parsing=kmer_parsing, data=1)
        random_seed = 101

        kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
        fold = 1
        for split_index in kf.split(dfdata_set):
            print("Start training/testing fold {}".format(fold))
            print()
            dftraining_set = dfdata_set.iloc[split_index[0]]
            dftest_set = dfdata_set.iloc[split_index[1]]

            dftraining_set = dftraining_set[["sentences", "bound"]]
            dftest_set = dftest_set[["sentences", "bound"]]

            test_set = dftest_set.to_dict("records")
            training_set = dftraining_set.to_dict("records")

            vector_models = train_vectors(training_set, categories, file_name, args, logger)

            w2v_hold_true, w2v_hold_pred, w2v_hold_prob = test_vectors(test_set, vector_models, logger)

            fpr, tpr, _ = roc_curve(w2v_hold_true, w2v_hold_prob, pos_label=1)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = metrics.precision_recall_curve(w2v_hold_true, w2v_hold_prob, pos_label=1)
            avg_prc = metrics.average_precision_score(w2v_hold_true, w2v_hold_prob)

            # ONLY the best model will be saved
            if not results_cv:
                for vector_class, model in enumerate(vector_models):
                    model_file = file_name + "_cat_{}".format(str(vector_class)) + ".w2v"
                    model.save(os.path.join(workingdir, model_file))
                    logger.info(model_file)
            elif results_cv[-1].get("roc_auc") < roc_auc:
                for vector_class, model in enumerate(vector_models):
                    model_file = file_name + "_cat_{}".format(str(vector_class)) + ".w2v"
                    model.save(os.path.join(workingdir, model_file))

            results_cv.append({"false_positive_rate": fpr,
                               "true_positive_rate": tpr,
                               "roc_auc": roc_auc,
                               "precision": precision,
                               "recall": recall,
                               "avg_prec": avg_prc})
            fold += 1

        print("Done training {} times".format(cv))
        helpers.save_plot_prc(results_cv, prc_figure_file, dataset_name)
        helpers.save_plot_roc(results_cv, roc_figure_file, dataset_name)
    else:
        print("You shouldn't be here!")
        sys.exit(1)

    return results_cv

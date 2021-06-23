# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' % Introduction
# ' Some of the code below is still under active development

# ' TODO: include other distance options for distance_kmer_vec(),
# ' TODO: to allow users to pick, e.g., pearson correlation

# ' ## Required libraries
# + name = 'import_libraries', echo=False
import os
import time
import random
import pandas as pd
import numpy as np

from scipy import interp
from sklearn.metrics import auc

from pybedtools import BedTool
from pyfaidx import Fasta
from collections import namedtuple, deque, OrderedDict
from itertools import product, islice

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as style


# + name = 'set_BLACK_LIST', echo=False
def set_BLACK_LIST(datafile):
    """
    Params
    -------
    datafile: `string`
        global variable
    """
    data_ranges_df = BedTool(datafile).to_dataframe()
    data_ranges_df = data_ranges_df.infer_objects()
    black_list = BedTool.from_dataframe(data_ranges_df)
    global BLACK_LIST
    BLACK_LIST = black_list

# + name = 'set_CONTROL_WINDOW', echo=False
def set_CONTROL_WINDOW(window):
    """
    Params
    -------
    fastafolder: `string`
        global variable
    """
    global CONTROL_WINDOW
    CONTROL_WINDOW = window

# + name = 'set_CONTROL_EVALUATION', echo=False
def set_CONTROL_EVALUATION(evaluate):
    """
    Params
    -------
    fastafolder: `string`
        global variable
    """
    global CONTROL_EVALUATION
    CONTROL_EVALUATION = evaluate


# + name = 'set_FASTA_FOLDER', echo=False
def set_FASTA_FOLDER(fastafolder):
    """
    Params
    -------
    fastafolder: `string`
        global variable
    """
    global FASTA_FOLDER
    FASTA_FOLDER = fastafolder


# + name = 'set_PREFIX', echo=False
def set_PREFIX(prefix):
    """
    Params
    -------
    prefix: `string`
        global variable
    """
    global PREFIX
    PREFIX = prefix

# + name = 'get_FASTA_FOLDER', echo=False
def get_FASTA_FOLDER():
    """
    Return
    -------
    FASTA_FOLDER: `string`
        global variable
    """
    return FASTA_FOLDER


# + name = 'get_PREFIX', echo=False
def get_PREFIX():
    """
    Return
    -------
    PREFIX: `string`
        global variable
    """
    return PREFIX


# + name = 'get_CONTROL_WINDOW', echo=False
def get_CONTROL_WINDOW():
    """
    Return
    -------
    PREFIX: `string`
        global variable
    """
    return CONTROL_WINDOW


# + name = 'get_CONTROL_EVALUATION', echo=False
def get_CONTROL_EVALUATION():
    """
    Return
    -------
    PREFIX: `string`
        global variable
    """
    return CONTROL_EVALUATION


# + name = 'get_BLACK_LIST', echo=False
def get_BLACK_LIST():
    """
    Params
    -------
    datafile: `string`
        global variable
    """
    return BLACK_LIST


# + name = 'get_chrom_names_list', echo=False
def get_chrom_names_list(datafile):
    """Return the names of the chromosomes to be queries as a list
    
    Parameters
    ----------
    datafile: `str` with a path that includes the chromosome lenghts
    
    Return
    ----------
    chrom_queries: `list`

    """
    data_ranges = BedTool(datafile).to_dataframe()
    data_ranges = data_ranges.infer_objects()
    chrom_queries = data_ranges["chrom"].unique().tolist()
    return chrom_queries


# + name = 'get_random_fragment', echo=False
def get_random_fragment(end_coord, n, start_coord=0, randint=101):
    """Divide a region (start, end) into n fragments of equal size and randomly return one of the fragments.
    This is an itertool recipe
    Parameters
    ----------
    start_coord: `int`, optional
        start coord of the fragment (0, by default).
    end_coord: `int`
        final coord of the fragment
    n: `int`
        the number of times that the given fragment will be divided.
    
    Returns
    -------
    rndchunk: `namedtuple`
        namedtuple rndchunk with components
        start (`int`)
        end (`int`)
    """
    random.seed(randint)  # for reproducibility
    len_fragment = end_coord - start_coord
    len_window = int(len_fragment / n)
    fragment = namedtuple("fragment", "start end")
    fragment_list = []
    while len(fragment_list) < n:
        fragment_list.append(fragment(int(start_coord), int(start_coord + len_window)))
        start_coord += len_window
    rndchunk = random.choice(fragment_list)
    return rndchunk


# + name = 'slice_fragment', echo=False
def slide_fragment(start_coord, end_coord, wsize=3, stepsize=0):
    """Sliding through a region between coords (start_coord, end_coord) to yield fragments size wsize, and step size stepsize
    This is an itertool recipe
    Parameters
    ----------
    start_coord: (`int`)
        start coord of the fragment 
    end_coord: (`int`)
        final coord of the fragment
    wsize: `int`, optional, (3, by default)
        size of the fragment to be yield
    stepsize: `int`, optional, (0, by default - non-overlapping fragments)
        positions to be overlapping between the fragments
    
    Yields
    ----------
    chunk: `namedtuple`
        Result chunk with components:
        start (`int`)
        end (`int`)
    """
    fragment = namedtuple("fragment", "start end")
    d = deque(maxlen=wsize)
    it = iter(range(start_coord, end_coord))
    for i in range(stepsize):
        d.append(next(it))
    while True:
        for i in range(wsize - stepsize):
            try:
                d.append(next(it))
            except StopIteration:
                if i > 0:
                    astart = list(d)[wsize - stepsize - i:][0]
                    bend = list(d)[wsize - stepsize - i:][-1] + 1
                    yield fragment(int(astart), int(bend))
                return
        cstart = list(d)[0]
        dend = list(d)[-1] + 1
        yield fragment(int(cstart), int(dend))


# + name = 'make_new_token', echo=False
def make_new_token(kmer):
    """Write a collapsed k-mer and k-mer reverse complementary as a newtoken
    
    Parameters
    ----------
    kmer: `string`
        String of the alphabet ATCG
        
    Return
    ----------
    newtoken: `string`
        A new token is a new string that combines the input string and its reverser complement "ATnTA"
    """
    kmer = str(kmer).upper()
    newtoken = "n".join(sorted([kmer, kmer.translate(str.maketrans("TAGC", "ATCG"))[::-1]]))
    return newtoken


# + name = 'create_kmer_set', echo=False
def create_kmer_set(ksize):
    """Write all possible new tokens of the alphabet ATCG of size k
    
    Parameters
    ----------
    ksize: `int`
        the size k of desired string (k-mer)
    
    Return
    ----------
    uniq_kmers: `list`
        alphabetically sorted unique new tokens
    """
    kmerSet = set()
    newtokenSet = set()
    nucleotides = ["A", "C", "G", "T"]
    kmerall = product(nucleotides, repeat=ksize)
    for i in kmerall:
        kmer = ''.join(i)
        kmerSet.add(kmer)
    uniq_kmers = sorted(list(kmerSet))
    for kmer in uniq_kmers:
        newtoken = make_new_token(kmer)
        newtokenSet.add(newtoken)
    uniq_newtokens = sorted(list(newtokenSet))
    return uniq_newtokens


# + name = 'get_kmer_vec', echo=False
def get_kmer_vec(sequence, ksize=2):
    """Write a dictionary with relative frequencies of observed k-mers for a given k in a sequence
    stored as dictionary keys are all the unique new tokens for a given k
    
    Parameters
    ----------
    sequence: `string`
        expecting string of size >= ksize
    ksize: `int`, optional
        k is the size of desired k-mer, (2, by default)
    
    Return
    ----------
    vec: `list` of floats
        the values of the dictionary with the frequencies of k-mers, always in the same order!
    """
    seq = str(sequence).upper()
    kmerset = create_kmer_set(ksize)
    finalstart = (len(seq) - ksize) + 1
    allkmers = [seq[start:(start + ksize)] for start in range(0, finalstart)]
    tokens = [make_new_token(kmer) for kmer in allkmers if len(kmer) == ksize and "N" not in kmer]
    nucvec = OrderedDict()
    for kmer in kmerset:
        nucvec[kmer] = 0.0
    kmer_count = dict(pd.value_counts(tokens))
    for key, value in kmer_count.items():
        if key in nucvec.keys():
            nucvec[key] = value / len(allkmers)
    vec = list(nucvec.values())
    return vec


# + name = 'distance_kmer_vec', echo=False
def distance_kmer_vec(queryvec, targetvec):
    """Calculate the cosine similarity between two vectors
    
    Parameters
    ----------
    queryvec, targetvec: `list` with floats 
        corresponding to relative frequencies to compare
        
    Return
    ----------
    cos_sim: `float`
        cosine similarity
    """
    import operator
    import math
    cos_sim = math.nan
    try:
        prod = sum(map(operator.mul, queryvec, targetvec))
        len1 = math.sqrt(sum(map(operator.mul, queryvec, queryvec)))
        len2 = math.sqrt(sum(map(operator.mul, targetvec, targetvec)))
        denom = (len1 * len2)
        if denom > 0:
            cos_sim = prod / denom
        else:
            raise RuntimeWarning
    except RuntimeWarning:
        pass
    return cos_sim


# + name = 'get_meth_vec', echo=False
def get_meth_vec(sequence):
    """Write a list with relative frequencies of trinucleotide motifs important for the methylation context
    CG 3-mer motifs (CCC CGG CCG CGC GCC GGG GCG GGC) 
    CHG motifs (CAG, CTG, and CCG)
    CHH motifs (CAA, CAT, CAC, CTT, CTA, CTC, CCC, CCA, and CCT)
    
    Parameters
    ----------
    sequence: `string`

    Return
    ----------
    vec: `list` of floats
        frequencies for the 3-mers related to methylation context. values are always in the same order!
    """
    kmersize = 3
    uniq_kmers = list(set(
        ["CGG", "CGC", "GCC", "GCG", "GGC", "CAG", "CTG", "CCG", "CAA", "CAT", "CAC", "CTT", "CTA", "CTC", "CCC", "CCA",
         "CCT"]))
    newtokenSet = set()
    for kmer in uniq_kmers:
        newtoken = make_new_token(kmer)
        newtokenSet.add(newtoken)
    kmerset = sorted(list(newtokenSet))
    seq = str(sequence).upper()
    finalstart = (len(seq) - kmersize) + 1
    allkmers = [seq[start:(start + kmersize)] for start in range(0, finalstart)]
    tokens = [make_new_token(kmer) for kmer in allkmers if len(kmer) == kmersize and "N" not in kmer]
    nucvec = OrderedDict()
    for kmer in kmerset:
        nucvec[kmer] = 0.0
    kmer_count = dict(pd.value_counts(tokens))
    for key, value in kmer_count.items():
        if key in nucvec.keys():
            nucvec[key] = value / len(allkmers)
    vec = list(nucvec.values())
    return vec

# + name = 'split_train_test', echo=False
def split_train_test(dataset, train_percent=.7, seed=42):
    """split inputfile in two dataframes for train, and test a kmergrammar model
    
    Parameters
    ----------
    inputfile: `dataset`  
        is a pandas dataframe
    train_percent: `double`,  (.6, by default)
    seed: `double`,  (42, by default)
    
    Return
    ----------
    trndf, valdf, tstdf: `pd.DataFrame`
        trndf corresponding to 70%, and tstdf corresponding to 30% (fraction) of the entries in the input
    """
    np.random.seed(seed)
    categories = [0, 1]
    trndf = []
    tstdf = []
    for cat in categories:
        catdata = dataset[dataset['bound'] == cat]
        perm = np.random.permutation(catdata.index)
        m = len(catdata.index)
        train_end = int(train_percent * m)
        trndf.append(catdata.loc[perm[:train_end]])
        tstdf.append(catdata.loc[perm[train_end:]])

    return pd.concat(trndf), pd.concat(tstdf)


# + name = 'split_train_validate_test', echo=False
def split_train_validate_test(dataset, train_percent=.6, validate_percent=.2, seed=42):
    """split inputfile in three dataframes for train, validate, and test a kmergrammar model
    
    Parameters
    ----------
    inputfile: `dataset`  
        is a pandas dataframe
    train_percent: `double`,  (.6, by default)
    validate_percent: `double`,  (.2, by default)
    seed: `double`,  (42, by default)
    
    Return
    ----------
    trndf, valdf, tstdf: `pd.DataFrame`
        trndf corresponding to 60%, valdf corresponding to 20%, and tstdf corresponding to 20% (fraction) of the entries in the input
    """
    np.random.seed(seed)
    categories = [0, 1]
    trndf = []
    valdf = []
    tstdf = []
    for cat in categories:
        catdata = dataset[dataset['bound'] == cat]
        perm = np.random.permutation(catdata.index)
        m = len(catdata.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        trndf.append(catdata.loc[perm[:train_end]])
        valdf.append(catdata.loc[perm[train_end:validate_end]])
        tstdf.append(catdata.loc[perm[validate_end:]])

    return pd.concat(trndf), pd.concat(valdf), pd.concat(tstdf)

# + name = 'write_dbtables', echo=False
def write_dbtables(database, validation):
    """write the splitted data
    
    Parameters
    ----------
    dbfile: result from `generate_input_db(args, logger, run_id)`
    alldatatable: `string`
    ntables: `int`,  (2, by default)
    """
    import sqlite3

    conn = sqlite3.connect(database)
    table = "dataset"
    query = "SELECT * from {} ;".format(table)
    df = pd.read_sql_query(query, conn)
    chrom_query = df["chrom"].unique().tolist()
    conn.close()
    for idx_chr in chrom_query:
        chrom_df = df[df["chrom"] == idx_chr]
        if not validation:
            trndf, tstdf = split_train_test(chrom_df)  # split 70% train, 30% test
            conn = sqlite3.connect(database)
            trndf.to_sql("train", conn, if_exists="append", index=False)
            tstdf.to_sql("test", conn, if_exists="append", index=False)
            conn.close()
        else:
            conn = sqlite3.connect(database)
            trndf, valdf, tstdf = split_train_validate_test(chrom_df)  # split 60% train, 20% validation, 20% test
            trndf.to_sql("train", conn, if_exists="append", index=False)
            tstdf.to_sql("test", conn, if_exists="append", index=False)
            valdf.to_sql("validation", conn, if_exists="append", index=False)
            conn.close()


# + name = 'get_dna_string_for_df', echo=False
def get_dna_string_for_df(dataframe):
    """not_split if user doesn't want to split the data into two dataframes
    
    Parameters
    ----------
    inputfile: `string`  
        is a path to a file separated with tabs, and columns corresponding to ['chrom', 'start', 'end', 'name','score']
    
    Return
    ----------
    dataset: `pd.DataFrame`
        
    """
    dataset = dataframe[["chrom", "start", "end", "name", "score"]]
    dataset["dna_string"] = dataset.apply(get_dna_string, axis=1)
    dataset["bound"] = 1
    return dataset


# + name = 'get_dna_string', echo=False
def get_dna_string(row):
    """Return a sequence corresponding to a genomic interval from a given fasta file
    
    Parameters
    ----------
    row:
    
    Return
    ----------
    sequence: `string`
        from a fasta file
    """
    GENOME = Fasta(os.path.join(get_FASTA_FOLDER(), get_PREFIX() + ".fa"), as_raw=True, sequence_always_upper=True)
    chrom = str(row[0])
    start = int(row[1])
    end = int(row[2])
    sequence = GENOME[chrom][start:end]
    return sequence


# + name = 'get_chrom_length', echo=False
def get_chrom_length(chrom):
    """Reads a chrom.info file and returns value in column 2 for an entry in the file
    
    Parameters
    ----------
    chrom: `str`
        an alphanumeric storing the name of a chromosome
        it will be casted to str to deal with 'Mt', 'Pt', 'chr1' and other formats for chromosome name
    
    Return
    ----------
    length: `int` 
        return a integer with the length of the chromosome
    """
    index_file = os.path.join(get_FASTA_FOLDER(), get_PREFIX() + ".fa.fai")
    chrom = str(chrom)
    fasta_index = pd.read_csv(index_file, dtype=str, sep="\t", header=None, usecols=[0, 1],
                              names=["chrom", "chrom_length"])
    length = int(fasta_index.loc[fasta_index["chrom"] == chrom]["chrom_length"].item())
    return length


# + name = 'get_chrom_length_df', echo=False
def get_chrom_length_df(chrom):
    """Reads a chrom.info file and returns value in column 2 for an entry in the file
    
    Parameters
    ----------
    chrom: `str`
        an alphanumeric storing the name of a chromosome
        it will be casted to str to deal with 'Mt', 'Pt', 'chr1' and other formats for chromosome name
    
    Return
    ----------
    length: `int` 
        return a integer with the length of the chromosome
    """
    index_file = os.path.join(get_FASTA_FOLDER(), get_PREFIX() + ".fa.fai")
    chrom = str(chrom)
    fasta_index = pd.read_csv(index_file, dtype=str, sep="\t", header=None, usecols=[0, 1],
                              names=["chrom", "chrom_length"])
    return fasta_index


# + name = 'get_dataset_ranges', echo=False
def get_dataset_ranges(bedfile):
    """Write a bedfile with all the regions to be omitted from the control
    
    Parameters
    ----------
    bedfile: `string``
        a path to a bed file
    
    Return
    ----------
    dataset: `BedTool`
    """
    dataset = BedTool(bedfile)
    return dataset


# + name = 'get_gc_perc', echo=False
def get_gc_perc(seq):
    """Write a dictionary with a control for a given query region
    
    Parameters
    ----------
    seq: `string`
    
    Return
    ----------
    freq_GC: `int`
        integer with the relative frequency of GCs in seq
    """
    freq_GC = int(round((sum([1.0 for nucl in seq if nucl in ["G", "C"]]) / len(seq)) * 100))
    return freq_GC


# + name = 'evaluate_control', echo=False
def evaluate_control(hit_seq, query, control_id=1):
    """write a boolean to accept or reject the candidate control region
    
    Parameters
    ----------
    hit_seq: a string, the candidate control region
    query: a string, the query region
    control_id: `int`, optional
        key from the dictionary with options to evaluate hit and query (1, by default)
    
    Return
    ----------
    keep_hit: `boolean`
        `True` if hit_seq pass the evaluation condition, `False` otherwise.
    """
    evaluate_dict = {1: "gc", 2: "di", 3: "met"}
    # freq = 0.98
    evaluate = evaluate_dict.get(control_id)
    keep_hit = False
    if evaluate == "di":
        cos_hit_query = distance_kmer_vec(get_kmer_vec(hit_seq), get_kmer_vec(query))
        if cos_hit_query >= 0.95:
            keep_hit = True
        else:
            keep_hit = False
    elif evaluate == "gc":
        if get_gc_perc(hit_seq) > get_gc_perc(query) - 1 and get_gc_perc(hit_seq) < get_gc_perc(query) + 1:
            keep_hit = True
        else:
            keep_hit = False
    elif evaluate == "met":
        cos_hit_query = distance_kmer_vec(get_meth_vec(hit_seq), get_meth_vec(query))
        if cos_hit_query >= 0.90:
            keep_hit = True
        else:
            keep_hit = False
    return keep_hit


# + name = 'windows', echo=False
def windows(input_values, length=2, overlap=0):
    """Slice an iterable between windows of given length and given overlap
    
    Parameters
    ----------
    iterable: 
    length: `int`, optional
    overlap: `int`, optional
    
    Return
    ----------
    results_array: `list`
    """
    results_array = []
    iterable = iter(input_values)
    window = list(islice(iterable, length))
    while len(window) == length:
        results_array.append(window)
        window = window[length - overlap:]
        window.extend(islice(iterable, length - overlap))
    if window:
        results_array.append(window)
    return results_array


# + name = 'group_generator', echo=False
def group_generator(data):
    """Yield groups out of a pandas dataframe
    
    Parameters
    ----------
    data: `pd.DataFrame()`
    length: `int`, optional
    overlap: `int`, optional
    
    Return
    ----------
    results_array: `list`
    """
    for name, group in data.groupby("query_id"):
        yield group


# + name = 'onehot_encoded_DNA_record', echo=False
def onehot_encoded_DNA_record(dnastring, matrix, record):
    """Write an string from the alphabet ATCG as onehotencoded matrix
    
    Parameters
    ----------
    dnastring: 
    matrix: 
    record: 
    
    Return
    ----------
    matrix
    """
    dnadict = {"A": 0, "C": 1, "G": 2, "T": 3}
    position = 0
    for dnaletter in dnastring:
        if dnaletter in dnadict:
            matrix[record, position + dnadict.get(dnaletter)] = 1.0
        else:  # i.e., 'N' or any other character
            matrix[record, position + 0] = 0.25
            matrix[record, position + 1] = 0.25
            matrix[record, position + 2] = 0.25
            matrix[record, position + 3] = 0.25
        position = position + 5

    return matrix


# + name = 'rev_comp_encoded_record', echo=False
def rev_comp_encoded_record(dnastring, matrix, record):
    """Write an string from the alphabet ATCG as special onehotencoded for the DNA double strand 
    
    Parameters
    ----------
    
    Return
    ----------
    """
    return matrix


# + name = 'label_scaled', echo=False
def label_scaled(label, matrix, record, scaler):
    """Scaled-normalized labels for regression models
    
    Parameters
    ----------
    label
    matrix
    record
    scaler (should this be saved)
    
    #normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    Return
    ----------
    """
    return matrix


# + name = 'label_binary_class', echo=False
def label_binary_class(label, matrix, record):
    """Labels for binary classification models
    
    Parameters
    ----------

    
    Return
    ----------
    matrix
    """
    if label == 1:
        matrix[record, 0] = 0.0
        matrix[record, 1] = 1.0
    else:
        matrix[record, 0] = 1.0  # everything else is not-bound
        matrix[record, 1] = 0.0
    return matrix


# + name = 'get_run_id', echo=False
def get_run_id():
    """Generates unique identifier
    
    Parameters
    ----------
    
    Return
    ----------
    
    """
    run_id = str(int(time.time()))
    return run_id


# + name = 'save_plot_prc', echo=False
def save_plot_prc(results_cv, figure_file, name):
    """Make plot for precission recall curve
    
    Parameters
    ----------
    precision: precision
    recall: recall
    avg_prec: avg_prec
    figure_file: figure_file
    name: name
    
    """
    style.use('fivethirtyeight')
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.title("Precision Recall Curve - {}".format(name))

    if len(results_cv) > 1:
        for idx, model in enumerate(results_cv, start=1):
            plt_label = "Fold {} Precision Avg = {}".format(idx, round(model.get("avg_prec"), 2))
            plt.plot(model.get("recall"), model.get("precision"), "b", label=plt_label)
    elif len(results_cv) == 1:
        for model in results_cv:
            plt_label = "Precision Avg = {}".format(idx, round(model.get("avg_prec"), 2))
            plt.plot(model.get("recall"), model.get("precision"), "b", label=plt_label)

    plt.legend(loc="best", fontsize="small")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall (sensitivity)")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(figure_file)


# + name = 'save_plot_roc', echo=False
def save_plot_roc(results_cv, figure_file, name):
    """Make plot for roc_auc
    
    Parameters
    ----------
    false_positive_rate: false_positive_rate
    true_positive_rate: true_positive_rate
    roc_auc: roc_auc
    figure_file: figure_file
    name: name
    
    """
    style.use('fivethirtyeight')
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.title("Receiver Operating Characteristic - {}".format(name))

    if len(results_cv) > 1:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for idx, model in enumerate(results_cv, start=1):
            plt_label = "CV{}_AUC = {}".format(idx, round(model.get("roc_auc"), 2))
            plt.plot(model.get("false_positive_rate"), model.get("true_positive_rate"), "b", label=plt_label)
            tprs.append(interp(mean_fpr, model.get("false_positive_rate"), model.get("true_positive_rate")))
            tprs[-1][0] = 0.0
            aucs.append(model.get("roc_auc"))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color="b", label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
                 alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2, label=r'$\pm$ 1 std. dev.')

    elif len(results_cv) == 1:
        for model in results_cv:
            plt_label = "AUC = {}".format(idx, round(model.get("roc_auc"), 2))
            plt.plot(model.get("false_positive_rate"), model.get("true_positive_rate"), "b", label=plt_label)

    plt.legend(loc="best", fontsize="small")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate (1-specificity")
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.tight_layout()
    plt.savefig(figure_file)

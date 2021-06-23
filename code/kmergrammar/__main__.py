#!/usr/bin/env python3

# ' % kmergrammar
# ' % Maria Katherine Mejia Guerra mm2842
# ' % 15th May 2017

# ' # Introduction
# ' Some of the code below is still under active development

# ' ## Required libraries

# + name = 'import_libraries', echo=False
import os
import sys
import time
import argparse
import itertools
from argparse import Namespace

import pandas as pd

from kgmodel import model
from kgutils import helpers
from kgutils.report import setup_logger
from kgutils.helpers import get_run_id, get_chrom_length, get_CONTROL_EVALUATION, get_CONTROL_WINDOW
from kgutils.helpers import get_dna_string_for_df, slide_fragment, evaluate_control
from kgutils.helpers import write_dbtables, get_dna_string
from multiprocessing import Pool, Manager, cpu_count

from pybedtools import BedTool
from sklearn.utils import shuffle


class BreakOutOfALoop(Exception):
    """Base class for exceptions in this module."""
    pass


# Create manager object in module-level namespace
mgr = Manager()

# Then create a container of things that you want to share to
# processes as Manager.Namespace() object.
config = mgr.Namespace()


# + name = 'get_control_sequences', echo=False
# this function is parallelizable and requires the substraction ahead of time to avoid io issues with BedTools file

def get_list_control_ranges(data):
    """Build a list of extended regions around given ranges that doesn't overlap with any given range
    
    Parameters
    ----------
    data: `pd.DataFrame()` 
        Likely coming from pd.DataFrame() it should contain ["chrom", "start", "end", "dna_string", "score", "bound"]
    
    Return
    ----------
    results: `list` of `pd.DataFrame()`
        contain ["chrom", "start", "end", "start_query", "end_query", "query_id", "string_query", "score_query", "runid_query"]
    """
    diameter = get_CONTROL_WINDOW()
    results = []
    black_list = config.black_list
    for index, row in data.iterrows():
        # get query coords and query sequence
        chrom = row.chrom
        start_query = row.start
        end_query = row.end

        # to find a control sequence around the query region get the search radious
        radious = int(((diameter - (end_query - start_query)) / 2) // 1)

        # get coords for the control region
        if start_query - radious < int(0):
            start_control_range = 0
        else:
            start_control_range = start_query - radious
        if end_query + radious > get_chrom_length(int(chrom)):
            end_control_range = get_chrom_length(int(chrom))
        else:
            end_control_range = end_query + radious

        ctrl_coords = "\t".join([str(chrom), str(start_control_range), str(end_control_range)])

        # remove positive and query regions from control region
        ctrlBed = BedTool(ctrl_coords, from_string=True)
        diffbed = ctrlBed.subtract(black_list)
        diffdf = diffbed.to_dataframe()
        diffdf["start_query"] = row.start
        diffdf["end_query"] = row.end
        diffdf["query_id"] = index
        diffdf["name"] = row.name
        diffdf["string_query"] = row.dna_string
        diffdf["score_query"] = row.score
        diffdf["runid_query"] = row.run_id
        results.append(diffdf)
    return results


def get_control_sequences_in_window(data):
    """Find a control region for a given query in a window around the query
    
    Parameters
    ----------
    data: `pd.DataFrame()` 
        should contain ["chrom", "start", "end", "start_query", "end_query", "query_id", "string_query", "score_query", "runid_query"]
    
    Return
    ----------
    list of hits: `list` of `dict`
        `dict` contains 
        chrom: `str` (alphanumeric)
        start: `int`
        end: `int`
        dna_string: `str`
        score: `float`
        bound: `int` 0 or 1
    """
    from interruptingcow import timeout
    import random
    randint = 101
    control_id = get_CONTROL_EVALUATION()
    chunk_results = {}
    control_category = int(0)
    for name, diffdf in data.groupby("query_id"):
        shuffle(diffdf,
                random_state=randint)  # these are bed intervals #random avoid bias towards the beginning of the range
        hit_in_radious = {}
        try:
            with timeout(1, exception=RuntimeError):
                randint = random.randint(1, 101)
                for diff in diffdf.itertuples():
                    chrom = diff.chrom
                    query = diff.string_query
                    len_query = len(query)
                    overlap = int((len(query) / 2) // 1)
                    if not hit_in_radious:
                        hitdb_in_radious = [val for val in slide_fragment(diff.start, diff.end, wsize=len_query,
                                                                                  stepsize=overlap)]
                        shuffle(hitdb_in_radious,
                                random_state=randint)  # decreases the chances for collision among control regions
                        for hit_range in hitdb_in_radious:
                            hit_seq = get_dna_string([chrom, hit_range.start, hit_range.end])
                            if len(hit_seq) == len_query and evaluate_control(hit_seq, query,
                                                                                      control_id=control_id):
                                hit_in_radious = {"chrom": chrom,
                                                  "start": int(hit_range.start),
                                                  "end": int(hit_range.end),
                                                  "name": diff.name,
                                                  "dna_string": hit_seq,
                                                  "score": diff.score_query,
                                                  "bound": control_category,
                                                  "runid_query": diff.runid_query,
                                                  "query_id": diff.query_id}
                                chunk_results[name] = hit_in_radious
                                raise BreakOutOfALoop
        except (BreakOutOfALoop, RuntimeError):
            pass
        if not hit_in_radious:
            hit_in_chrom = {"chrom": chrom,
                            "start": int(diff.start_query),
                            "end": int(diff.end_query),
                            "name": diff.name,
                            "dna_string": diff.string_query,
                            "score": diff.score_query,
                            "bound": int(1),
                            "runid_query": diff.runid_query,
                            "query_id": diff.query_id}
            chunk_results[name] = hit_in_chrom
    return list(chunk_results.values())


def get_control_sequences_in_chrom(data):
    """Find a control region for a given query in a window within the same chromosome
    
    Parameters
    ----------
    data: `pd.DataFrame()` 
        should contain ["chrom", "start", "end", "dna_string", "score", "bound"]
    
    Return
    ----------
    list of hits: `list` of `dict`
        `dict` contains 
        chrom: `str` (alphanumeric)
        start: `int`
        end: `int`
        dna_string: `str`
        score: `float`
        bound: `int` 0 value
    """
    from interruptingcow import timeout
    import random
    black_list = config.black_list
    length = config.length
    random.seed(101)  # to help with reproducibility
    control_id = helpers.get_CONTROL_EVALUATION()
    control_category = int(0)
    hard_results = {}
    for index, row in data.iterrows():
        # hit = False
        hit_in_chrom = {}
        # get query coords and query sequence
        chrom = row.chrom
        query = row.dna_string
        len_query = len(query)
        overlap = int((len(query) / 2) // 1)
        try:
            with timeout(2, exception=RuntimeError):
                randint = random.randint(1, 101)
                chrom_range = helpers.get_random_fragment(length, 100, randint=randint)
                ctrl_coords = "\t".join([str(chrom), str(chrom_range.start), str(chrom_range.end)])

                # remove positive regions from control region
                ctrlBed = BedTool(ctrl_coords, from_string=True)
                diffbed = ctrlBed.subtract(black_list)
                diffdf = diffbed.to_dataframe()
                shuffle(diffdf, random_state=randint)

                for diff in diffdf.itertuples():
                    hitdb_in_chrom = [val for val in
                                      slide_fragment(diff.start, diff.end, wsize=len_query, stepsize=overlap)]
                    shuffle(hitdb_in_chrom, random_state=randint)
                    for hit_range in hitdb_in_chrom:
                        hit_seq = get_dna_string([chrom, hit_range.start, hit_range.end])
                        if not hit_in_chrom and len(hit_seq) == len_query and evaluate_control(hit_seq, query,
                                                                                                       control_id=control_id):
                            hit_in_chrom = {"chrom": chrom,
                                            "start": int(hit_range.start),
                                            "end": int(hit_range.end),
                                            "name": row.name,
                                            "dna_string": hit_seq,
                                            "score": row.score,
                                            "bound": control_category,
                                            "runid_query": row.runid_query,
                                            "query_id": row.query_id}
                            hard_results[row.query_id] = hit_in_chrom
                            raise BreakOutOfALoop

        except (RuntimeError, BreakOutOfALoop):
            continue
    return list(hard_results.values())


def mp_get_control_sequences_in_window(data, num_processes):
    """Parallelize the use of the function get_control_sequences_in_window
    
    Parameters
    ----------
    data: list of `pd.DataFrame()` 
    num_processes: `int`
        expected the maximun number of processes to be multiprocessing.cpu_count() - 1
    
    Return
    ----------
    results: `tuple`
        resultdf["bound"] == 0 and resultdf["bound"] == 1
    """
    examples = data.query_id.unique()
    # print(data.head())
    chunk_size = int(len(examples) / num_processes)
    chunk_ranges = helpers.windows(examples, length=chunk_size, overlap=0)
    chunks = [data[data["query_id"].isin(chunk_ids)] for chunk_ids in chunk_ranges]
    try:
        pool = Pool(processes=num_processes)
        results_parallel = [pool.apply_async(get_control_sequences_in_window, args=(c,)) for c in chunks]
    finally:
        pool.close()
        pool.join()
    pool.terminate()
    result_list = list(itertools.chain.from_iterable([p.get() for p in results_parallel]))  # flattening
    resultdf = pd.DataFrame(result_list)
    return resultdf[resultdf["bound"] == 0], resultdf[resultdf["bound"] == 1]


def mp_get_control_sequences_in_chrom(data, num_processes):
    """Parallelize the use of the function get_control_sequences_in_chrom
    
    Parameters
    ----------
    data: list of `pd.DataFrame()` 
    num_processes: `int`
    expected the maximum number of processes to be multiprocessing.cpu_count() - 1
    
    Return
    ----------
    results: `pd.DataFrame()`
    """
    try:
        chunk_size = int(data.shape[0] / num_processes) + 1
        chunks = [data.iloc[i:i + chunk_size, :] for i in range(0, data.shape[0], chunk_size)]
        pool = Pool(processes=num_processes)
        results_parallel = pool.map(get_control_sequences_in_chrom, chunks)
    finally:
        pool.close()
        pool.join()
    pool.terminate()
    result_list = list(itertools.chain.from_iterable(results_parallel))
    hard_controls = pd.DataFrame(result_list)
    return hard_controls


def mp_get_control_ranges(data, num_processes):
    """Parallelize the use of the function get_control_ranges

    Parameters
    ----------
    data: list of `pd.DataFrame()`
    num_processes: `int`
    expected the maximum number of processes to be multiprocessing.cpu_count() - 1

    Return
    ----------
    results: `pd.DataFrame()`
    """
    try:
        chunk_size = int(data.shape[0] / num_processes)
        chunks = [data.iloc[i:i + chunk_size, :] for i in range(0, data.shape[0], chunk_size)]
        pool = Pool(processes=num_processes)
        results_parallel = pool.map(get_list_control_ranges, chunks)
    finally:
        pool.close()
        pool.join()
    pool.terminate()
    result_list = list(itertools.chain.from_iterable(results_parallel))
    control_ranges = pd.concat(result_list)
    return control_ranges


def evaluate_preprocessing_arguments(arguments, logger):
    """
    Parameters
    ----------
    arguments
    logger
    
    Return
    ----------
    dict
    """
    evaluation_dict = {"gc": [1, "GC CONTENT"],
                       "di": [2, "DINUCLEOTIDE CONTENT"],
                       "met": [3, "METHYLATION CONTEXT"]}

    basename = os.path.basename(arguments.datafile)
    fileprefix = os.path.splitext(basename)[0]

    datafile = os.path.join(os.path.dirname(arguments.datafile), fileprefix + '.bed')

    if arguments.control not in evaluation_dict.keys():
        logger.info("INPUT OPTION {} TO SELECT CONTROLS DOESN'T EXIST".format(arguments.control))
        logger.info("USING DEFAULT OPTION GC CONTENT TO SELECT CONTROLS")
        print("INPUT OPTION {} TO SELECT CONTROLS DOESN'T EXIST - SET TO DEFAULT GC CONTENT".format(arguments.control))
        helpers.set_CONTROL_EVALUATION(1)
    else:
        helpers.set_CONTROL_EVALUATION(evaluation_dict.get(arguments.control)[0])
        logger.info("USING INPUT OPTION {} TO SELECT CONTROLS".format(arguments.control))
        dbdir = os.path.join(os.path.dirname(arguments.datafile), arguments.control)
        dbfile = os.path.join(dbdir, fileprefix + '.db')

    if not os.path.exists(dbdir):
        os.makedirs(dbdir)
    elif os.path.exists(dbfile):
        logger.info("DATABASE FILE {} ALREADY EXIST".format(os.path.abspath(dbfile)))
        print("Fatal: DATABASE FILE {} ALREADY EXIST, CAN'T OVERWRITE".format(os.path.abspath(dbfile)))
        sys.exit(1)

    if not os.path.exists(datafile):
        logger.info("DATAFILE {} DOESN'T EXIST".format(datafile))
        print("DATAFILE {} DOESN'T EXIST".format(datafile))
        sys.exit(1)
    else:
        chrom_query = helpers.get_chrom_names_list(datafile)
        chrom_query_length = [helpers.get_chrom_length(chrom) for chrom in chrom_query]
        max_window_size = int(min(chrom_query_length) / 200)
        min_window_size = 50
        if arguments.window <= max_window_size or arguments.window >= min_window_size:
            helpers.set_CONTROL_WINDOW(arguments.window)
        else:
            logger.info("INPUT WINDOW SIZE IS OUTSIDE OF ACCEPTED BOUNDARIES")
            print("Warning: INPUT WINDOW SIZE IS OUTSIDE OF ACCEPTED BOUNDARIES")
            if int(max_window_size) > 125000:
                helpers.set_CONTROL_WINDOW(125000)
            else:
                helpers.set_CONTROL_WINDOW(max_window_size)

    if not os.path.exists(os.path.join(arguments.fastadir, arguments.prefix + ".fa")):
        if not os.path.exists(os.path.join(arguments.fastadir, arguments.prefix + ".fasta")):
            logger.info("FASTA FILE {} DOESN'T EXIST".format(os.path.join(arguments.fastadir, arguments.prefix + ".fa")))
            print("Fatal: FASTA FILE {} DOESN'T EXIST".format(os.path.join(arguments.fastadir, arguments.prefix + ".fa")))
            sys.exit(1)
    elif not os.path.exists(os.path.join(arguments.fastadir, arguments.prefix + ".fa.fai")):
        if not os.path.exists(os.path.join(arguments.fastadir, arguments.prefix + ".fasta.fai")):
            logger.info("FASTA_FAI FILE {} DOESN'T EXIST".format(os.path.join(arguments.fastadir, arguments.prefix + ".fa.fai")))
            print("Fatal: FASTA_FAI FILE {} DOESN'T EXIST".format(os.path.join(arguments.fastadir, arguments.prefix + ".fa.fai")))
            sys.exit(1)
    else:
        logger.info("FASTA_FAI FILE {}".format(os.path.join(arguments.fastadir, arguments.prefix + ".fa.fai")))
        logger.info("FASTA FILE {} ".format(os.path.join(arguments.fastadir, arguments.prefix + ".fa")))

    max_cpu = cpu_count() - 1
    if arguments.process > max_cpu:
        num_processes = cpu_count() - 1
        print("Wrong option -process {}. Maximum number of available processors is {}".format(arguments.process, max_cpu))
        logger.info("Warning: Wrong option -process {}".format(arguments.process))
        logger.info("using {}/{} of available processors".format(num_processes, max_cpu))
    else:
        num_processes = arguments.process
        print("Using {}/{} of available processors".format(num_processes, max_cpu))
        logger.info("using {}/{} number of processors".format(num_processes, max_cpu))

    return datafile, dbfile, num_processes


def generate_input_db(arguments, logger, run_id):
    """Write sqlite3 file for all the input data
    
    Parameters
    ----------
    args: result from `parser.parse_args()`
        args.datafile: `string`
        args.fastadir: `string`
        args.prefix: `string`
        args.window: `int`
        args.control: `string`
        args.strata: `boolean`
        args.validation: `boolean`
        
    logger: `logger`
    run_id: `string` 
        unique identifier of the session
        :rtype: object
    """
    import sqlite3

    helpers.set_PREFIX(arguments.prefix)
    helpers.set_FASTA_FOLDER(arguments.fastadir)
    datafile, dbfile, num_processes = evaluate_preprocessing_arguments(arguments, logger)
    table = "dataset"
    data_ranges = BedTool(datafile).to_dataframe()
    data_ranges = data_ranges.infer_objects()
    chrom_query = data_ranges["chrom"].unique().tolist()
    logger.info("BUILDING DATABASE {}".format(dbfile))
    logger.info("{} chromosomes in input file".format(len(chrom_query)))

    print("BUILDING DATABASE {}".format(dbfile))
    for idx_chr in chrom_query:
        chr_start_time = time.time()
        length = helpers.get_chrom_length(idx_chr)
        chrom_df = data_ranges[data_ranges["chrom"] == idx_chr]
        max_size = 5000
        logger.info("WRITING CONTROL FOR CHROMOSOME {} - STRATIFIED: {}".format(idx_chr, arguments.strata))

        # list of dicts -> df -> BedTool
        # TODO - take a pandas DF and make it into a list of dictionaries
        # TODO - try to pass a dictionary as an argument directly in the mp functions
        black_list = BedTool.from_dataframe(chrom_df)
        config.black_list = black_list
        config.length = length

        print("Reading {} regions for chromosome {}".format(chrom_df.shape[0], idx_chr))
        dataset_collection = [chrom_df.iloc[i:i + max_size, :] for i in range(0, chrom_df.shape[0], max_size)]

        for idx_dataset, dataset in enumerate(dataset_collection):
            print("Processing {}/{} from chromosome {}".format(idx_dataset + 1, len(dataset_collection), idx_chr))

            current_dataset = get_dna_string_for_df(dataset)  # fix method signature
            current_dataset["run_id"] = run_id
            current_dataset = current_dataset.infer_objects()
            logger.info("Processing {} ranges from chrom {}".format(dataset.shape[0], idx_chr))

            # got control regions substracted from black_list
            start_time = time.time()
            current_control_regions = mp_get_control_ranges(current_dataset, num_processes)
            elapsed_time = time.time() - start_time
            took_control_ranges = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            logger.info(
                "generate {} control ranges for {} input regions took {}".format(current_control_regions.shape[0], len(
                    current_control_regions.query_id.unique()), took_control_ranges))

            # easy controls for test data
            start_time = time.time()
            current_easy_controls, current_try_again = mp_get_control_sequences_in_window(current_control_regions,
                                                                                          num_processes)
            current_easy_controls = current_easy_controls[
                ["chrom", "start", "end", "name", "dna_string", "score", "bound"]]
            current_easy_controls = current_easy_controls.infer_objects()
            current_easy_controls["run_id"] = run_id
            total_results = current_easy_controls.shape[0] + current_try_again.shape[0]
            elapsed_time_chunk = time.time() - start_time
            took_parallel_chunk = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_chunk))
            logger.info("Processing {} examples took {}".format(total_results, took_parallel_chunk))
            logger.info(
                "We bring {} controls in radious and {} outside of radious".format(current_easy_controls.shape[0],
                                                                                   current_try_again.shape[0]))

            conn = sqlite3.connect(dbfile)
            current_dataset.to_sql(table, conn, if_exists="append", index=False)
            current_easy_controls.to_sql(table, conn, if_exists="append", index=False)
            conn.close()

            # hard results can also use multiprocess
            if current_try_again.shape[0] > 0:
                current_hard_controls = mp_get_control_sequences_in_chrom(current_try_again, num_processes)
                current_hard_controls = current_hard_controls[
                    ["chrom", "start", "end", "name", "dna_string", "score", "bound"]]
                current_hard_controls = current_hard_controls.infer_objects()
                current_hard_controls["run_id"] = run_id
                elapsed_time_hard = time.time() - start_time
                took_hard = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_hard))
                logger.info(
                    "hard_query took {} to bring {} test control regions".format(took_hard, len(current_hard_controls)))

                conn = sqlite3.connect(dbfile)
                current_hard_controls.to_sql(table, conn, if_exists="append", index=False)
                conn.close()

        elapsed_time_chr = time.time() - chr_start_time
        took_chr = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_chr))
        print("chromosome {} took {} to complete control regions".format(idx_chr, took_chr))
        print()

    # write tables for split data
    write_dbtables(dbfile, arguments.validation)

    con = sqlite3.connect(dbfile)
    cursor = con.cursor()
    cursor.execute("SELECT count(*) from {} WHERE bound = 1".format(table))
    test_1 = cursor.fetchall()
    cursor.execute("SELECT count(*) from {} WHERE bound = 0".format(table))
    test_0 = cursor.fetchall()
    con.close()
    logger.info("DATABASE READY {} bound = 1: {}, bound = 0: {}".format(dbfile,
                                                                        test_1[0][0], test_0[0][0]))
    total_input_regions = test_1[0][0]
    total_control_results = test_0[0][0]
    print("DATABASE READY {} bound = 1: {}, bound = 0: {}".format(dbfile,
                                                                  test_1[0][0], test_0[0][0]))
    logger.info("Done preprocessing {} input regions to generate {} control regions".format(total_input_regions,
                                                                                            total_control_results))


def evaluate_modeling_arguments():
    return True


def read_arguments(arguments=None):
    """The main routine in charge of parsing the user options and make the calls to the code
    Parameters
    ----------
    arguments
    """
    text = "A tool for training several machine learning architectures.py on DNA sequences"
    parser = argparse.ArgumentParser(prog="kmergrammar",
                                     description=text,
                                     usage="%(prog)s [options]")

    # two main commands preprocessing and model
    subparsers = parser.add_subparsers(help="kmergrammar sub-commands include:",
                                       dest="which")

    preprocessing_parser = subparsers.add_parser("preprocessing",
                                                 description="usage: python kmergrammar preprocessing\n" +
                                                             "-data test_NAM_peaks_m3\n" +
                                                             "-fasta ZmB73_AGPv4/\n" +
                                                             "-prefix Zea_mays.AGPv4.dna\n" +
                                                             "-w 10000\n" +
                                                             "-di -stratified")
    preprocessing_parser.add_argument("-data",
                                      action="store",
                                      dest="datafile",
                                      help="BED file",
                                      required=True)
    preprocessing_parser.add_argument("-fasta",
                                      action="store",
                                      dest="fastadir",
                                      help="directory for fasta and fai files",
                                      required=True)
    preprocessing_parser.add_argument("-prefix",
                                      action="store",
                                      dest="prefix",
                                      help="prefix of the fasta and fai files",
                                      required=True)
    preprocessing_parser.add_argument("-window",
                                      type=int,
                                      action="store",
                                      dest="window",
                                      default=10000,
                                      help="number of bases to extend around midpoint of each entry (DEFAULT: 10000)")
    preprocessing_parser.add_argument("-process",
                                      type=int,
                                      action="store",
                                      dest="process",
                                      default=4,
                                      help="number of processors for multiprocessing")
    group_control = preprocessing_parser.add_mutually_exclusive_group()
    group_control.add_argument("-gc",
                               action='store_const',
                               dest='control',
                               const='gc',
                               help="For each entry find a control region with similar GC content (DEFAULT)")
    group_control.add_argument("-di",
                               action='store_const',
                               dest='control',
                               const='di',
                               help="For each entry find a control region with similar dinucleotide content")
    group_control.add_argument("-met",
                               action='store_const',
                               dest='control',
                               const='met',
                               help="For each entry find a control region with similar methylation context")
    group_control.set_defaults(control='gc')

    feature_parser = preprocessing_parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument("-stratified",
                                dest="strata",
                                action="store_true",
                                help="To use the fifth column of datafile to generate stratified train/test")
    feature_parser.add_argument("-no-stratified",
                                dest="strata",
                                action="store_false",
                                help="Don't generate stratified train/test (DEFAULT)")
    feature_parser.set_defaults(strata=False)

    split_parser = preprocessing_parser.add_mutually_exclusive_group(required=False)
    split_parser.add_argument("-validation",
                              dest="validation",
                              action="store_true",
                              help="Generate tables train, test and validation")

    split_parser.add_argument("-no-validation",
                              dest="validation",
                              action="store_false",
                              help="Don't generate a validation table (Default)")
    split_parser.set_defaults(validation=False)

    model_parser = subparsers.add_parser("model",
                                         description="usage: python kmergrammar model\n" +
                                                     "-type bagkmer\n" +
                                                     "-param_file test.txt \n" +
                                                     "-train\n")

    model_parser.add_argument("-type",
                              action="store",
                              dest="type",
                              help="types of models currently available: bagkmer, vkmer, CNN, CNN_LSTM - REQUIRED",
                              required=True)

    model_parser.add_argument("-param_file",
                              action="store",
                              dest="paramfile",
                              help="configuration file with parameters - REQUIRED",
                              required=True)

    group_model = model_parser.add_mutually_exclusive_group(required=False)
    group_model.add_argument("-train",
                             action="store_const",
                             dest="action",
                             const="train",
                             help="Expect database (train table) to fit a model - DEFAULT")
    group_model.add_argument("-test",
                             action="store_const",
                             dest="action",
                             const="test",
                             help="Expect database (test table) and a trained model")
    group_model.add_argument("-classify",
                             action="store_const",
                             dest="action",
                             const="classify",
                             help="Expect a trained model to provide predictions on unlabeled data")
    group_model.set_defaults(action="train")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    args = read_arguments()  # type: Namespace
    if args.which == "preprocessing":
        run_id = get_run_id()
        logfile = "kmergrammar_preprocessing_" + run_id + ".txt"
        logger = setup_logger("kmergrammar_preprocessing", logfile, run_id)
        print("")
        print("k-mer grammar preprocessing run-id: {}".format(run_id))
        generate_input_db(args, logger, run_id)
    elif args.which == 'model':
        run_id = get_run_id()
        logfile = "kmergrammar_model_{}_{}.txt".format(str(args.type), run_id)
        logger = setup_logger("kmergrammar_model", logfile, run_id)
        print("")
        print("k-mer grammar {} model type: {}, run-id: {}".format(str(args.action), str(args.type), run_id))
        model.sort_model(args, logger, run_id)

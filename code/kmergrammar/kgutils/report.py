#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#' % kmergrammar kgmodel
#' % Maria Katherine Mejia Guerra mm2842
#' % 15th May 2017

#' # Introduction
#' Some of the code below is under active development

#' ## Required libraries
#+ name = 'import_libraries', echo=False
import os
import sys
import inspect
import logging
import warnings

def setup_logger(name, log_file, run_id, level=logging.DEBUG):
    """Star a logger to document the calls of the library
    Parameters
    ----------
    name: `string` 
        the namespace for the logging
    log_file: `string` 
        a filehandler to append log messages
    run_id: `string` 
        unique identifier of the session
    level: (optional), default=logging.DEBUG
        logging level for the message 
    
    Return
    ----------
    logger: `logger`
        the result of calling logging.getLogger()
    """
    formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    log_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace("kgutils", "logs")
    handler = logging.FileHandler(log_dir+"/"+log_file, mode="a+")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.info("kmergrammar run-id: {}".format(run_id))
    return logger 
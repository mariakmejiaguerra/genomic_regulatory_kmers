#' % kmergrammar
#' % Maria Katherine Mejia Guerra mm2842
#' % 15th May 2017

#' # Introduction
#' Some of the code below is under active development


#' ## Required libraries
#+ name = 'import_libraries', echo=False
import os
import sys
sys.path.insert(0, '/Users/mm2842/KmerGrammar/') #once the package is registered with pip, this line will be ommited?

from kmergrammar.kgutils import report
from kmergrammar.kgutils import helpers

import unittest

#check first parsing options

class TestPreprocessing(unittest.TestCase):
    #TODO add the report of the test to a log file
    id_session = helpers.get_run_id()
    logfile = 'logfile-test-kmergrammar-' + id_session + '.txt'
    logger = report.setup_logger('kmergrammar', logfile, id_session)
    
    def setUp(self):
        pass
    
    def test_split_test_train(self):
        """
        Parameters
        ----------
        
        Return
        ----------

        """
        helpers.PREFIX = 'test_long'
        helpers.FASTA_FOLDERS = os.path.dirname(__file__)
        datafile = os.path.join(helpers.FASTA_FOLDERS,'test_long.bed')
        train, test = helpers.split_train_test(datafile, strata=True)
        self.assertEqual(test.shape[0],300)
        self.assertEqual(train.shape[0],700)
    
    def test_search_control(self):
        """
        Parameters
        ----------
        
        Return
        ----------

        """
        self.assertEqual("","")
    
    def test_get_dna_string(self):
        """
        Parameters
        ----------
        
        Return
        ----------

        """
        self.assertEqual("","")
 

class TestModelVecKmer(unittest.TestCase):
    #TODO add the report of the test to a log file
    id_session = helpers.get_run_id()
    logfile = 'logfile-test-kmergrammar-' + id_session + '.txt'
    logger = report.setup_logger('kmergrammar', logfile, id_session)
    
    def setUp(self):
        pass
    
    def test_split_test_train(self):
        """
        Parameters
        ----------
        
        Return
        ----------

        """
        self.assertEqual("",700)
    
    def test_search_control(self):
        """
        Parameters
        ----------
        
        Return
        ----------

        """
        self.assertEqual("","")
    
    def test_get_dna_string(self):
        """
        Parameters
        ----------
        
        Return
        ----------

        """
        self.assertEqual("","")
    

if __name__ == '__main__':
    
    unittest.main()

===========
kmergrammar
===========

Introduction
------------

**Kmergrammar - grammar of the k-mers - **
Kmergrammar has been designed to explore functional genomic information, such as MNAse-seq, ATAC-seq, or ChIP-seq
aimed to extract sequence features that are signatures of the regions characterized for such biochemical assays.
Further, Kmergrammar allows sequence augmentation, from the adding of information such as
Genetic Variation (i.e., MAF scores), or DNA conservation (i.e., GERP scores)

Input data
-------------
Kmergrammar reads the data from a sqlite3 database with sequence data
Including at least tables ``train`` and ``test``, which can be generated using kmergrammar

If you don't want to use kmergrammar, to generate the database you can create your own sqlite3 database

For example, kmergrammar uses the following statement to create ``train`` table with:

  .. code-block:: sql

     CREATE TABLE IF NOT EXISTS "train" (
     "chrom" INTEGER,
     "start" INTEGER,
     "end" INTEGER,
     "name" TEXT,
     "score" INTEGER,
     "dna_string" TEXT,
     "class" INTEGER,
     "run_id" TEXT)

TODO:
Add option to generate a database from fasta files

Classification tasks
====================
To classify binary only

TODO:
Add multiclassifiers

Training
--------
If you want to train new data

Prediction
----------
To predict

Regression tasks
==================
TODO:
Add regressions

Quick start
-----------

We have tested  installation on

Contact, support and questions
------------------------------
We are interested in all comments on the package,
and the ease of use of installation and documentation.


Credits
-------
This package is written and maintained by Dr Katherine Mejia-Guerra, under supervision of Dr. Ed Buckler.
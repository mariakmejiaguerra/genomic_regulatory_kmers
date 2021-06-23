==========================
INSTALL Guide For kmergrammar
==========================

Prerequisites
=============

kmergrammar is verified to work on XX (X X.X). 

Python version 3.5.

Numpy_ (>=1.6). 

Cython_ (>=0.18) is an optional requirement to recompile ``.pyx`` files.

Tensorflow_ (>=1.8)

Scikitlearn_ (>=0.19.1)

matplotlib_

bedtools_ (>=2.25)

.. _Numpy: http://www.scipy.org/Download
.. _Cython: http://cython.org/
.. _Tensorflow: https://www.tensorflow.org/	
.. _Scikitlearn: http://scikit-learn.org/
.. _matplotlib: https://matplotlib.org/
.. _bedtools: http://bedtools.readthedocs.io/

Download source and data
========================
To download the source code from our github repository::

 $ git clone https://
 
To download data files that are used in our paper::

 $ wget https://XXXXX

Then, place the folder named "data" under the kmergrammar directory.
 
Configure environment variables
===============================

You need to add the downloaded location (in this example home directory; $HOME) to your ``PYTHONPATH`` and ``PATH`` environment variables.

PYTHONPATH
~~~~~~~~~~

You need to include the new value in your ``PYTHONPATH`` by
adding this line to your ``~/.bashrc``::

 $ export PYTHONPATH=$HOME/kmergrammar/:$PYTHONPATH

Then, type::

 $ source .bashrc

Or, re-login to your account.

PATH
~~~~

You'll also like to add a new value to your
PATH environment variable so that you can use the kmergrammar command line
directly::

 $ export PATH=$HOME/kmergrammar/bin/:$PATH

--
Katherine Mejia-Guerra <mm2842@cornell.edu>

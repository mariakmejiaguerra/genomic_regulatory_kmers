#!/usr/bin/env python3

from setuptools import setup, find_packages


DESCRIPTION = "A tool for training NLP inspired machine/deep " \
              "learning architectures on DNA sequences"

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name="kmergrammar",
      version="0.0.1",
      author="Katherine Mejia-Guerra",
      author_email="mm2842@cornell.edu",
      description="A tool for training NLP inspired machine/deep learning architectures on DNA sequences",
      long_description=readme(),
      long_description_content_type="text/markdown",
      download_url="https://bitbucket.org/bucklerlab/kmergrammar/src/master/",
      url="https://bitbucket.org/bucklerlab/kmergrammar/src/master/",
      packages=find_packages(),
      entry_points={'console_scripts': ['kmergrammar = kmergrammar.__main__:main']},
      classifiers=[
          "Natural Language :: English",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
          # "Topic :: DNA Processing :: DNA Linguistic",
          "Intended Audience :: Science/Research",
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Linux :: Ubuntu"],
      license='GPLv3',
      install_requires=['markdown', 'numpy', 'pandas', 'sqlalchemy', 'gensim', 'scikit-learn', 'matplotlib', 'boto3',
                        'pyfaidx', 'pybedtools', 'interruptingcow', 'tensorflow'],
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=['bin/'],
      include_package_data=True,
      zip_safe=False)

*******************
Installing desidlas
*******************

This document describes how to install the `desidlas`
repository.

Installing Dependencies
=======================

We have and will continue to keep the number of dependencies low.
There are a few standard packages that must be installed
and little more.

In general, we recommend that you use Anaconda or
*pip* for the majority of these installations.

Detailed installation instructions are presented below:

Dependencies
------------

astropath depends on the following list of Python packages.

We recommend that you use `Anaconda <https://www.continuum.io/downloads/>`_
to install and/or update these packages.

* `python <http://www.python.org/>`_ versions 3.7 or later
* `numpy <http://www.numpy.org/>`_ version 1.18 or later
* `astropy <http://www.astropy.org/>`_ version 4.2 or later
* `scipy <http://www.scipy.org/>`_ version 1.4 or later
* `pandas <https://pandas.pydata.org/>`_ version 0.25 or later

If you are using Anaconda, you can check the presence of these packages with::

	conda list "^python|numpy|astropy|scipy|pandas"

If the packages have been installed, this command should print
out all the packages and their version numbers.

Install via ``pip``
-------------------

To install the dependencies using `pip <https://pypi.org/project/pip/>`_:

 #. Download `requirements.txt <https://github.com/cosmodesi/desi-dlas/blob/main/desidlas/requirements.txt>`__.

 #. Install the dependencies::

        pip install -r requirements.txt

Note that this is a "system-wide" installation, and will
replace/upgrade any current versions of the packages you already have
installed.


Installing desidlas
===================

Presently, you must download the code from github::

	#go to the directory where you would like to install specdb.
	git clone https://github.com/cosmodesi/desi-dlas.git

From there, you can build and install with::

	cd desidlas
	python setup.py install  # or use develop


This should install the package and any related scripts.
Make sure that your PATH includes the standard
location for Python scripts (e.g. ~/anaconda/bin)

For Developers
==============

Do these for docs::

    pip install nbsphinx



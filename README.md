# agile21_abstaining
Reproducibility Package for our paper


# Section 1: Setting

(c) 2021 M. Werner

Information-optimal absatining - generate a synthetic dataset to showcase the algorithm on non-GDPR data

We have been applying our machinery to a collection of tweet objects obtained from the Twitter API. According
to Twitter regulations as well as to the situation with respect to privacy rights (right of users to delete data 
after we got them from the stream), we cannot publish the data.

Tweet IDs: But we are allowed to publish the involved tweet IDs. These are available in the synthetic dataset as the 
filenames. You will be able to download (most of) those tweets with your own API credentials using the given IDs.

Synthetic Dataset: The situation of the paper is roughly that a majority of the text objects assigned to commercial or
residential buildings by proximity is irrelevant for the building. But we believe a little bit is. Our synthetic problem
is therefore constructed from three text sources: 

- source 1: 90% of our data come from an English review corpus (Amazon Reviews, see [1])
- source 2: project Gutenberg version of Goethes Faust 
- source 3: project Gutenberg version of Dr. Faustus (Heinrich Heine)

The intuition is that our noise corpus (messages that do not correlate with our target) is a large one. Messages that contain
information about the current building are (1) from different sources of (2) with some semantic overlap.

We sample as follows: All clenaned documents are first turned into infite data streams from which a given (12) number of words is
extracted in each step. For all files in the original dataset, we create a novel file with the same file name according to the following rule:
in 90% of the cases, we sample the next 12 words from Amazon Reviews
in the remaining 10%, we follow our (here balanced) distribution of residential and commercial.


# Section 2: Overview

## Preflight

Before using this, extract la.tbz using ```tar -xjf la.tbz```. Don't extract the files in la-large-full, they are consumed in this form.


## abstaining.py

This file contains two aspects:
### Data Generator

The first section is related to generating the fake dataset as we can't share the Twitter data underpinning our research
for reasons of privacy and legal issues.

The second section is related to abstaining implementing the decision function (aCL) and the mutual information class (NI) which
is used in the optmization.

## 00_create_public_dataset.py

This will create a dataset as described above. You can't run it without providing the "reference" dataset (which can be the la.tgz dataset here) in another path and putting the path into this file (in the rglob line)

## 01_create_public_dataset_large.py

The same as above, but it generates a much larger non-balanced dataset from our confidential template dataset using public text.
Again, you can use it only with a template, take our public dataset as the template if you want to reproduce it. Move it somewhere and give it to the system

**It is important to manually compress the file generated. For encoding reasons, this has been pruned from the python code. run
```
bzip la-large-full/single-file.txt.bz2
```
## 02_single_classifiers.py

This implements the first experiment. It is based on the text mining documentation of scikit learn and just.

BEING DERIVED FROM A BSD LICENSE SOURCE, THIS FILE IS UNDER BSD LICENSE!
Compare cycler==0.10.0
joblib==1.0.1
kiwisolver==1.3.1
matplotlib==3.4.2
numpy==1.20.2
Pillow==8.2.0
pyparsing==2.4.7
python-dateutil==2.8.1
scikit-learn==0.24.2
scipy==1.6.3
six==1.16.0
sklearn==0.0
threadpoolctl==2.1.0
tqdm==4.60.0
cycler==0.10.0
joblib==1.0.1
kiwisolver==1.3.1
matplotlib==3.4.2
numpy==1.20.2
Pillow==8.2.0
pyparsing==2.4.7
python-dateutil==2.8.1
scikit-learn==0.24.2
scipy==1.6.3
six==1.16.0
sklearn==0.0
threadpoolctl==2.1.0
tqdm==4.60.0


## 03_abstain_la.py

Here, we add abstaining to the classifiers.

BEING DERIVED FROM A BSD LICENSE SOURCE, THIS FILE IS UNDER BSD LICENSE!
compare with https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


## 04_abstain_voting_lalarge.py

Here we abstain voting and abstaining from ensembles of abstaining classifiers. This does only work on the large yet unbalanced dataset.

BEING DERIVED FROM A BSD LICENSE SOURCE, THIS FILE IS UNDER BSD LICENSE!
compare with https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html



# Appendix

## Docker log including all versions

```
martin@werner:~/2021work/agile21_abstaining$ docker build -t agile21_abstaining .
Sending build context to Docker daemon  81.02MB
Step 1/3 : FROM python:3.7.10-buster
 ---> 3a781253f798
Step 2/3 : ADD requirements.txt /requirements.txt
 ---> 05b53dbfa16e
Step 3/3 : RUN pip3 install -r requirements.txt
 ---> Running in 5feef65e5e09
Collecting scipy
  Downloading scipy-1.6.3-cp37-cp37m-manylinux1_x86_64.whl (27.4 MB)
Collecting sklearn
  Downloading sklearn-0.0.tar.gz (1.1 kB)
Collecting tqdm
  Downloading tqdm-4.60.0-py2.py3-none-any.whl (75 kB)
Collecting matplotlib
  Downloading matplotlib-3.4.2-cp37-cp37m-manylinux1_x86_64.whl (10.3 MB)
Collecting numpy
  Downloading numpy-1.20.2-cp37-cp37m-manylinux2010_x86_64.whl (15.3 MB)
Collecting scikit-learn
  Downloading scikit_learn-0.24.2-cp37-cp37m-manylinux2010_x86_64.whl (22.3 MB)
Collecting pyparsing>=2.2.1
  Downloading pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
Collecting cycler>=0.10
  Downloading cycler-0.10.0-py2.py3-none-any.whl (6.5 kB)
Collecting python-dateutil>=2.7
  Downloading python_dateutil-2.8.1-py2.py3-none-any.whl (227 kB)
Collecting kiwisolver>=1.0.1
  Downloading kiwisolver-1.3.1-cp37-cp37m-manylinux1_x86_64.whl (1.1 MB)
Collecting pillow>=6.2.0
  Downloading Pillow-8.2.0-cp37-cp37m-manylinux1_x86_64.whl (3.0 MB)
Collecting six
  Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting joblib>=0.11
  Downloading joblib-1.0.1-py3-none-any.whl (303 kB)
Collecting threadpoolctl>=2.0.0
  Downloading threadpoolctl-2.1.0-py3-none-any.whl (12 kB)
Building wheels for collected packages: sklearn
  Building wheel for sklearn (setup.py): started
  Building wheel for sklearn (setup.py): finished with status 'done'
  Created wheel for sklearn: filename=sklearn-0.0-py2.py3-none-any.whl size=1316 sha256=ef71da7545f3df1bcc1336ebd3d93330ea076a8845b07fd2ce610ce4e1938c90
  Stored in directory: /root/.cache/pip/wheels/46/ef/c3/157e41f5ee1372d1be90b09f74f82b10e391eaacca8f22d33e
Successfully built sklearn
Installing collected packages: numpy, threadpoolctl, six, scipy, joblib, scikit-learn, python-dateutil, pyparsing, pillow, kiwisolver, cycler, tqdm, sklearn, matplotlib
Successfully installed cycler-0.10.0 joblib-1.0.1 kiwisolver-1.3.1 matplotlib-3.4.2 numpy-1.20.2 pillow-8.2.0 pyparsing-2.4.7 python-dateutil-2.8.1 scikit-learn-0.24.2 scipy-1.6.3 six-1.16.0 sklearn-0.0 thread
poolctl-2.1.0 tqdm-4.60.0
WARNING: Running pip as root will break packages and permissions. You should install packages reliably by using venv: https://pip.pypa.io/warnings/venv
Removing intermediate container 5feef65e5e09
 ---> d2a558e4c4c4
Successfully built d2a558e4c4c4
Successfully tagged agile21_abstaining:latest

```



## Synthetic Dataset Generator Demo

Without the original dataset, the synthetic data generators are a bit strange as you can only run them on the synthetic dataset I have given to you. But this is how you could proceed:
```
martin@werner:~/2021work/agile21_abstaining$cd input
martin@werner:~/2021work/agile21_abstaining/input$ tar -xjf ../la.tbz
martin@werner:~/2021work/agile21_abstaining/input$ cd ..
martin@werner:~/2021work/agile21_abstaining$ rm la -Rf
martin@werner:~/2021work/agile21_abstaining$ python3 00_create_public_dataset.py 
64531it [00:02, 23020.90it/s]
``


And for the large one, put the one you have into input as well. And then

```
martin@werner:~/2021work/agile21_abstaining$ rm -Rf la-large-full/
martin@werner:~/2021work/agile21_abstaining$ python3 01_create_public_dataset_large.py
1060232it [00:11, 94323.75it/s]

Congratulations.

Note that you must compress the single-file.txt in the folder la-large-full yourself.
Run (for example): bzip2 la-large-full/single-file.txt

```
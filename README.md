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

Before using this, extract la.zip. Don't extract the files in la-large-full, they are consumed in this form.


## abstaining.py

This file contains two aspects:
### Data Generator

The first section is related to generating the fake dataset as we can't share the Twitter data underpinning our research
for reasons of privacy and legal issues.

The second section is related to abstaining implementing the decision function (aCL) and the mutual information class (NI) which
is used in the optmization.

## 00_create_public_dataset.py

This will create a dataset as described above. You can't run it without providing the "reference" dataset (which can be the la.tgz dataset here) in another path and putting the path into this file (in the rglob line)

01_create_public_dataset_large.py

The same as above, but it generates a much larger non-balanced dataset from our confidential template dataset using public text.
Again, you can use it only with a template, take our public dataset as the template if you want to reproduce it. Move it somewhere and give it to the system

**It is important to manually compress the file generated. For encoding reasons, this has been pruned from the python code. run
```
bzip la-large-full/single-file.txt.bz2
```
02_single_classifiers.py

This implements the first experiment. It is based on the text mining documentation of scikit learn and just.

THIS FILE IS UNDER BSD LICENSE!

03_abstain_la.py

Here, we add abstaining to the classifiers.

04_abstain_voting_lalarge.py

Here we abstain voting and abstaining from ensembles of abstaining classifiers. This does only work on the large yet unbalanced dataset.

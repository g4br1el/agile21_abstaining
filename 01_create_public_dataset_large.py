"""
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

source 1: 90% of our data come from an English review corpus (Amazon Reviews, see [1])
source 2: project Gutenberg version of Goethes Faust 
source 3: project Gutenberg version of Dr. Faustus (Heinrich Heine)

The intuition is that our noise corpus (messages that do not correlate with our target) is a large one. Messages that contain
information about the current building are (1) from different sources of (2) with some semantic overlap.

We sample as follows: All clenaned documents are first turned into infite data streams from which a given (12) number of words is
extracted in each step. For all files in the original dataset, we create a novel file with the same file name according to the following rule:
in 90% of the cases, we sample the next 12 words from Amazon Reviews
in the remaining 10%, we follow our (here balanced) distribution of residential and commercial.

"""

import re;
import json
import numpy as np;
import random
import os;
from tqdm import tqdm
from pathlib import Path
    

from abstaining import get_datasets;

import bz2;

if __name__=="__main__":
    np.random.seed(42)
    random.seed (42)

    drfaust, faust, reviews = get_datasets()
    
    os.mkdir("la-large-full")
    out = open("la-large-full/single-file.txt","w")
    lineage = open("01_create_public_dataset_large_lineage.json","w")
    for line in tqdm(bz2.BZ2File("input/single-file.txt.bz2","r")): # note that you don't have this file. But you can take the one in this repository if you like to test this script and or the parameters
        record = dict()
        line = line.decode()
        if np.random.random() < 0.9:
            whichone = reviews
            record["choice"]="reviews"
        else:
             if "commercial" == line.split(" ")[1]:
                record["choice"] = "faust"
                text = next(faust)
             else:
                record["choice"] = "drfaust"
                text = next(drfaust)
        line = line.split(" ")
        string =  (" ".join([line[0], line[1], next(whichone)]))
        record["text"]= string
        lineage.write(json.dumps(record)+"\n")
        out.write(string + "\n")
        
    print(""" 
Congratulations. 
                  
Note that you must compress the single-file.txt in the folder la-large-full yourself.
Run (for example): bzip2 la-large-full/single-file.txt
             """)

import pandas as pd
import numpy as np
import nltk
import os
import matplotlib.pyplot as plt
from nltk.tokenize import *
from itertools import groupby

ROOT_DIR = os.path.abspath(os.curdir)
df = pd.read_csv(str(ROOT_DIR + "\\myapp\\data\\summerdata_original.csv"))

# for loop stopper
counter = 0
tag_list = []
pos_list = []
tag_freq_dict = {}
for row in df['title_orig']:
    tokenized = nltk.word_tokenize(row)
    tagged = nltk.pos_tag(tokenized)
    for sentence in tagged:
        for tag, pos in tagged:
            tag_list.append(tag)
            pos_list.append(pos)

unique_tagged_list = list(dict.fromkeys(tag_list))
for tag in unique_tagged_list:
    tag_freq_dict.update({tag: tag_list.count(tag)})

data = tag_freq_dict
names = list(data.keys())
values = list(data.values())
plt.bar(names, values)
plt.show()
# for index, row in df.iterrows():
#     print(row)
#     counter += 1

import pandas as pd
import numpy as np
import os
import json
import glob
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold

df1 = pd.read_csv('../../input/external_data/ruddit/Dataset/ruddit_with_text.csv')
comment_list = df1['txt'].values
id_list = df1['comment_id'].values
id_list1 = []
offensiveness_score_list = (df1["offensiveness_score"].values + 1.) / 2.
print(offensiveness_score_list.mean(), offensiveness_score_list.min(), offensiveness_score_list.max())
data_dict = {}
for i in tqdm(range(len(id_list))):
    data_dict[id_list[i]] = {'text': comment_list[i], 'labels': offensiveness_score_list[i]}
    if isinstance(comment_list[i], str) and comment_list[i] != '[deleted]':
        id_list1.append(id_list[i])
print(len(id_list), len(id_list1))

if not os.path.exists('split1/'):
    os.makedirs('split1/')
with open('split1/train_id_list1.pickle', 'wb') as f:
    pickle.dump(id_list1, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('split1/data_dict.pickle', 'wb') as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)




import pandas as pd
import numpy as np
import os
import json
import glob
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold

df1 = pd.read_csv('../../input/external_data/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv')
id_list = df1['id'].values
id_list1 = []
comment_list = df1['comment_text'].values
toxic_list = df1['toxicity'].values
severe_toxic_list = df1['severe_toxicity'].values
obscene_list = df1['obscene'].values
threat_list = df1['threat'].values
insult_list = df1['insult'].values
identity_hate_list = df1['identity_attack'].values
sexual_list = df1['sexual_explicit'].values
data_dict = {}
for i in tqdm(range(len(id_list))):
    if isinstance(comment_list[i], str):
        data_dict[id_list[i]] = {'text': comment_list[i], 'labels': np.array([toxic_list[i], severe_toxic_list[i], obscene_list[i], threat_list[i], insult_list[i], identity_hate_list[i], sexual_list[i]])}
        id_list1.append(id_list[i])
print(len(id_list), len(id_list1))

if not os.path.exists('split1/'):
    os.makedirs('split1/')
with open('split1/train_id_list1.pickle', 'wb') as f:
    pickle.dump(id_list1, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('split1/data_dict.pickle', 'wb') as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)




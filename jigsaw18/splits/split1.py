import pandas as pd
import numpy as np
import os
import json
import glob
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold

df = pd.read_csv('../../input/validation_data.csv')
more_toxic_list = df['more_toxic'].values
less_toxic_list = df['less_toxic'].values
toxic_set = set(list(more_toxic_list) + list(less_toxic_list))
toxic_list = sorted(list(toxic_set))
print(len(more_toxic_list), len(less_toxic_list), len(toxic_list))

df1 = pd.read_csv('../../input/external_data/jigsaw-toxic-comment-classification-challenge/train.csv')
id_list = df1['id'].values
id_list1 = []
comment_list = df1['comment_text'].values
toxic_list = df1['toxic'].values
severe_toxic_list = df1['severe_toxic'].values
obscene_list = df1['obscene'].values
threat_list = df1['threat'].values
insult_list = df1['insult'].values
identity_hate_list = df1['identity_hate'].values
data_dict = {}
for i in tqdm(range(len(id_list))):
    data_dict[id_list[i]] = {'text': comment_list[i], 'labels': np.array([toxic_list[i], severe_toxic_list[i], obscene_list[i], threat_list[i], insult_list[i], identity_hate_list[i]])}
    if comment_list[i] not in toxic_set:
        id_list1.append(id_list[i])
print(len(id_list), len(id_list1))

if not os.path.exists('split1/'):
    os.makedirs('split1/')
with open('split1/train_id_list.pickle', 'wb') as f:
    pickle.dump(id_list, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('split1/train_id_list1.pickle', 'wb') as f:
    pickle.dump(id_list1, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('split1/data_dict.pickle', 'wb') as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)




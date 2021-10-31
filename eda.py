
#%%
import pandas as pd
import os

#%%

DATA_PATH = './data'

#%%

df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

#%%

df.head()

#%%

df['abstract_len'] = df['abstract'].str.len()

#%%

df['title_len'] = df['title'].str.len()

#%%

df['abstract_wc'] = df['abstract'].apply(lambda x: len(x.split(' ')))

#%%

df['title_wc'] = df['title'].apply(lambda x: len(x.split(' ')))

#%%

df.describe()

#%%

#%%

#%%

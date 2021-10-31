
#%%

import pandas as pd
import os
from tqdm import tqdm

from transformers import pipeline # , AutoModelForTokenClassification, AutoTokenizer

DATA_PATH = './data'

#%%

train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

#%%

summarize = pipeline('summarization')

#%%

MAX_LEN = 16 # TODO FROM TOKENIZER
test_titles = []
for abstract in tqdm(test['abstract']):
    summarized_text = summarize(abstract, max_length=MAX_LEN)
    test_titles.append(summarized_text[0]['summary_text'])

#%%

test_pred = test.copy()
test_pred['title'] = [v for v in test_titles] # [0]['summary_text']
test_pred.head(15)

#%%

test_pred.to_csv('test_pred.csv', index=None)

#%%

from create_submission import generate_csv

generate_csv('test_pred.csv', 'submission_v5.csv', os.path.join(DATA_PATH, 'vocs.pkl'))

#%% 

# Sentiment analysis pipeline
# summarize = pipeline('summarization')

# Question answering pipeline, specifying the checkpoint identifier
# pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

#%%


# pipe(df['abstract'][0], max_length=70)




# %%

# df['title'][0]

# %%

MAX_LEN = 16
offset = 160
for i in range(10):
    pred = summarize(train['abstract'][offset+i], max_length=MAX_LEN,\
        num_beans=6, length_penalty=2.0, early_stopping=True)
    pred = pred[0]['summary_text']
    orig = train['title'][offset+i]
    print(f'ORIG: {orig}\nPRED: {pred}\n')



# %%

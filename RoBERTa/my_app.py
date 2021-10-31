
# %%



import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow_hub as hub
import tf_sentencepiece

# %%

text_generator = hub.Module(
    'https://tfhub.dev/google/bertseq2seq/roberta24_bbc/1')
input_documents = ['This is text from the first document.',
                   'This is text from the second document.']
output_summaries = text_generator(input_documents)


# %%

body = """
two types of room temperature detectors of terahertz laser radiation have been developed which allow in an all-electric manner to determine the plane of polarization of linearly polarized radiation and the ellipticity of elliptically polarized radiation, respectively. the operation of the detectors is based on photogalvanic effects in semiconductor quantum well structures of low symmetry. the photogalvanic effects have sub-nanosecond time constants at room temperature making a high time resolution of the polarization detectors possible.
"""
model = Summarizer()
model(body, num_sentences=1)

# %%

from summarizer import TransformerSummarizer
model = TransformerSummarizer(transformer_type='GPT2', \
    transformer_model_key='gpt2')





# %%

import pandas as pd
import os
from tqdm import tqdm

DATA_PATH = '../data'

#%%

train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))


#%%

# MAX_LEN = 18 # TODO FROM TOKENIZER
test_titles = []
for abstract in tqdm(test['abstract']):
    # summarized_text = summarize(abstract, max_length=MAX_LEN)
    summary = model(abstract, num_sentences=1)
    test_titles.append(summary)

#%%

test_pred = test.copy()
test_pred['title'] = [v.strip() for v in test_titles] # [0]['summary_text']
test_pred.head(15)

#%%

test_pred.to_csv('test_pred.csv', index=None)

#%%

# test_pred['title_l'] = test_pred['title'].apply(lambda v: len(v))
# test_pred['title_l'].describe()
test_pred['title']  = test_pred['title_l'].apply(lambda v: v if v else 'empty')

# %%

from create_submission import generate_csv

generate_csv('test_pred.csv', 'submission_v4_lib2.csv', os.path.join(DATA_PATH, 'vocs.pkl'))


# %% 

from summarizer import TransformerSummarizer
model = TransformerSummarizer(transformer_type='GPT2', \
    transformer_model_key='gpt2')

# %%

offset = 150
for i in range(10):
    pred = model(train['abstract'][offset+i], \
        num_sentences=1, min_length=40, max_length=120)
    pred = pred # [0]['summary_text']
    orig = train['title'][offset+i]
    print(f'ORIG: {orig}\nPRED: {pred}\n')

# %%

# %%

# %%

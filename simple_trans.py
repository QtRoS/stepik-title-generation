import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = [
    ["one", "1"],
    ["two", "2"],
]

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = [
    ["three", "3"],
    ["four", "4"],
]

eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 10,
    "train_batch_size": 2,
    "num_train_epochs": 10,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 15,
    "manual_seed": 4,
    "is_decoder": False,
}

encoder_type = "roberta"

model = Seq2SeqModel(
    encoder_type,
    "roberta-base",
    "openai-gpt",
    args=model_args,
    use_cuda=True,
)

model.train_model(train_df)

results = model.eval_model(eval_df)

print(model.predict(["five"]))


model1 = Seq2SeqModel(
    encoder_type,
    encoder_decoder_name="outputs",
    args=model_args,
    use_cuda=True,
)

print(model1.predict(["five"]))

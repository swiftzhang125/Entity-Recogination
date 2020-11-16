export MAX_LEN=128
export EPOCHS=10
export TRAIN_BATCH_SIZE=4
export VALIDATION_BATCH_SIZE=4
export LR=3e-5
export BASE_MODEL='bert'
export BERT_PATH='bert-base-uncased'
export TOKENIZER_PATH='bert-base-uncased'
export DF_PATH='../input/ner_dataset.csv'
export META_PATH='../input/meta.bin'
export MODEL_PATH='../input/model.bin'

python train.py
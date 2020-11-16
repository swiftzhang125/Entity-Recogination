import transformers

MAX_LEN = 128
MODEL_PATH = '../input/model.bin'
META_PATH = '../input/meta.bin'
BASE_MODEL = 'bert'
BERT_PATH = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)
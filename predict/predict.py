import torch
import config
import joblib
import sys
sys.path.append('../src')
from model_dispatcher import MODEL_DISPATCHER
from dataset import EntityDataset

import warnings

warnings.filterwarnings('ignore')

DEVICE = 'cuda'

def predict():
    meta_data = joblib.load(config.META_PATH)

    enc_pos = meta_data['enc_pos']
    enc_tag = meta_data['enc_tag']

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    sentence = '''
    Alice will go to China this Saturday! Her father works in WHO
     
    .
    '''
    tokenized_sentence = config.TOKENIZER.encode(sentence)
    sentence = sentence.split()
    print('\n')
    print('sentence', sentence)
    print('tokenized_sentence', tokenized_sentence)

    test_dataset = EntityDataset(
        words=[sentence],
        pos=[[0] * len(sentence)],
        tags=[[0] * len(sentence)],
        tokenizer=config.TOKENIZER,
        max_len=config.MAX_LEN
    )

    model = MODEL_DISPATCHER[config.BASE_MODEL](bert_path=config.BERT_PATH,
                                         num_tag=num_tag,
                                         num_pos=num_pos
                                         )

    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(DEVICE)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(DEVICE).unsqueeze(0)
        tag, pos, _ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )

        print(
            enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )


if __name__ == '__main__':
    predict()

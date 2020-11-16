import torch
import os
from tqdm import tqdm
import joblib
import transformers
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from model_dispatcher import MODEL_DISPATCHER
from process_data import process_data
from dataset import EntityDataset
from sklearn import model_selection
import warnings

warnings.filterwarnings('ignore')

DEVICE = 'cuda'

MAX_LEN = int(os.environ.get('MAX_LEN'))
EPOCHS = int(os.environ.get('EPOCHS'))
TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
VALIDATION_BATCH_SIZE = int(os.environ.get('VALIDATION_BATCH_SIZE'))
LR = float(os.environ.get('LR'))

BASE_MODEL = os.environ.get('BASE_MODEL')
BERT_PATH = os.environ.get('BERT_PATH')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH')

DF_PATH = os.environ.get('DF_PATH')
META_PATH = os.environ.get('META_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')


def train_loop_fn(dataloader, model, optimizer, device, scheduler):
    model.train()
    fin_loss = 0
    for d in tqdm(dataloader, total=len(dataloader)):
        for k, v in d.items():
            d[k] = v.to(device)
        optimizer.zero_grad()
        _, _, loss = model(**d)
        loss.backward()
        optimizer.step()
        scheduler.step()
        fin_loss += loss.item()

    return fin_loss / len(dataloader)


def eval_loop_fn(dataloader, model, device):
    model.eval()
    fin_loss = 0
    for d in tqdm(dataloader, total=len(dataloader)):
        for k, v in d.items():
            d[k] = v.to(device)

        _, _, loss = model(**d)
        fin_loss += loss.item()

    return fin_loss / len(dataloader)

def run():
    sentences, pos, tag, enc_pos, enc_tag = process_data(DF_PATH)

    meta_data = {
        'enc_pos': enc_pos,
        'enc_tag': enc_tag
    }

    joblib.dump(meta_data, META_PATH)

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    (
        train_sentences,
        valid_sentences,
        train_pos,
        valid_pos,
        train_tag,
        valid_tag,
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=2020, test_size=0.1)

    tokenizer = transformers.BertTokenizer.from_pretrained(TOKENIZER_PATH, do_lower_case=True)

    train_dataset = EntityDataset(
        words=train_sentences,
        pos=train_pos,
        tags=train_tag,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = EntityDataset(
        words=valid_sentences,
        pos=valid_pos,
        tags=valid_tag,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        num_workers=4
    )

    model = MODEL_DISPATCHER[BASE_MODEL](bert_path=BERT_PATH,
                                         num_tag=num_tag,
                                         num_pos=num_pos
                                         )
    model.to(DEVICE)

    # parameters_optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    parameters_optimizer = [
        {
            'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.001,
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        }
    ]

    optimizer = AdamW(parameters_optimizer, lr=LR)
    num_training_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE * EPOCHS)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_loss = np.inf
    for epoch in range(EPOCHS):
        train_loss = train_loop_fn(train_dataloader, model, optimizer, DEVICE, scheduler)
        valid_loss = eval_loop_fn(valid_dataloader, model, DEVICE)

        print(f'Train_loss = {train_loss}, Valid_loss = {valid_loss}')

        if valid_loss < best_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_loss = valid_loss

if __name__ == '__main__':
    run()










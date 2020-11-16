import models

MODEL_DISPATCHER = {
    'bert': models.EntityModel
}

if __name__ == '__main__':
    print(MODEL_DISPATCHER['bert'](bert_path='bert-base-uncased', num_tag=8, num_pos=8))

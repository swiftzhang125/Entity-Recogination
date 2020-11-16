import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
import joblib

def process_data(data_path):
    df = pd.read_csv(data_path, encoding='latin-1')
    df.loc[:, 'Sentence #'] = df['Sentence #'].fillna(method='ffill')

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, 'POS'] = enc_pos.fit_transform(df['POS'])
    df.loc[:, 'Tag'] = enc_tag.fit_transform(df['Tag'])

    sentences = df.groupby('Sentence #')['Word'].apply(list).values
    pos = df.groupby('Sentence #')['POS'].apply(list).values
    tag = df.groupby('Sentence #')['Tag'].apply(list).values

    return sentences, pos, tag, enc_pos, enc_tag

if __name__ == '__main__':
    sentences, pos, tag, enc_pos, enc_tag = process_data('../input/ner_dataset.csv')
    print('sentence', sentences[0])
    print('pos', pos[0])
    print('tag', tag[0])

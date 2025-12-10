import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from typing import List
import pandas as pd
import re

PATH = './dataset/IMDB Dataset.csv'

# from tensorflow.keras.layers import TextVectorization
# MAX_SEQUENCE_LENGTH = 800
# VECTORIZER = TextVectorization(output_sequence_length=MAX_SEQUENCE_LENGTH)


def regex(data : str):
    data = re.sub(r'<.*?>', '', data)  
    data = re.sub(r'[^a-zA-Z\s]', '', data)   
    data = data.lower()
    data = data.strip()

    return data


def loading_and_preprocessing():
    df = pd.read_csv(PATH)

    print(f"Data Details : {df.describe()}")
    print(f"Null Sentiments Values {len(df[pd.isnull(df['sentiment'])])}")
    print(f"Null Review Values {len(df[pd.isnull(df['review'])])}")

    df.dropna(inplace=True)
    df.fillna({'sentiment' : '', 'review' : ''}, inplace=True)

    df['sentiment'] = df['sentiment'].apply(regex)
    df['review'] = df['review'].apply(regex)

    df['feedback'] = df['sentiment'].apply(lambda x : 1 if x == 'positive' else 0)
    
    return df

def data_preparation(dataframe : List[str], vectorizer):

    # unique_words = VECTORIZER.adapt(dataframe)
    # unique_words_length = len(VECTORIZER.get_vocabulary())
    # print(f"Length of Unique Words: {unique_words_length}")

    sequences = vectorizer(dataframe)
    # print(f"Shape of input sequences : {sequences.get_shape()}")
    return sequences
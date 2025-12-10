from tensorflow.keras.saving import load_model
import pickle
from tensorflow.keras.layers import TextVectorization
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

MODEL_PATH = './model/classifier.keras'
TOKENIZER_PATH = './model/tokenizer.pkl'

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as f:
    config = pickle.load(f)

vocabulary = config.pop('vocabulary', None)
tokenizer = TextVectorization.from_config(config)
if vocabulary:
    tokenizer.set_vocabulary(vocabulary)

def predict_value(text : str):
    seq = tokenizer([text]).numpy()
    output = model.predict(seq)

    print(output)

predict_value("Dhurandar has flashes of clever storytelling, but they’re buried under uneven pacing.The cast brings solid energy, even when the script doesn’t fully back them up.Some jokes work nicely, while others feel forced or mistimed.Visually, the film tries bold choices that sometimes pay off and sometimes distract.It’s an entertaining watch in parts, but far from consistently satisfying.")
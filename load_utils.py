import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import tensorflow as tf
from keras.layers import Layer
import streamlit as st
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


def load_pretrained_models():

    with open("models/model_DecisionTree_0.1.0.bin", 'rb') as file:
        model_dt = pickle.load(file)

    with open("models/model_GradientBoost_0.1.0.bin", 'rb') as file:
        model_gb = pickle.load(file)

    with open("models/model_LinearSVC_0.1.0.bin", 'rb') as file:
        model_svc = pickle.load(file)

    with open("models/model_LogisticRegression_0.1.0.bin", 'rb') as file:
        model_lr = pickle.load(file)

    with open("models/model_RandomForrest_0.1.0.bin", 'rb') as file:
        model_rf = pickle.load(file)

    return model_dt, model_gb, model_svc, model_lr, model_rf


def load_pretrained_text_vectorizer():

    with open('models/text_vectorizer_model_0.1.0.bin', 'rb') as file:
        text_vec = pickle.load(file)

    return text_vec


def pre_process_text(text):

    st = stopwords.words('english')
    port_stemmer = PorterStemmer()
    
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    stemmed_words = [port_stemmer.stem(word) for word in text if word not in st]
    text = ' '.join(stemmed_words)

    return text


def label_conv(s):
    if (s == 0):
        return "Fake News"
    else:
        return "Real News"
    

# The functions below are all used by the second tab, for the Advanced Attention Neural Network Model. 

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, features)
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1],),
                                 initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, time_steps, features)
        # compute scores
        scores = tf.tensordot(inputs, self.W, axes=1)  # (batch_size, time_steps)
        weights = tf.nn.softmax(scores, axis=1)  # (batch_size, time_steps)
        # weighted sum
        weighted = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)
        return weighted

    def get_config(self):
        base_config = super(AttentionLayer, self).get_config()
        return base_config
    

def pretty_model_summary(model):
    layers = []
    units = [(None, 60), (None, 60, 128), (None, 60, 128), (None, 60, 128), (None, 60, 128), (None, 128), (None, 128), (None, 256),  (None, 128), (None, 128), (None, 64), (None, 64), (None, 1)]
    
    for index, layer in enumerate(model.layers):

        layers.append({
            "Name": layer.name,
            "Type": layer.__class__.__name__,
            "Output Shape": str(units[index]),
            "Param #": layer.count_params()
        })

    df = pd.DataFrame(layers)
    st.dataframe(df)


# --------------------------
# 2) Utility: clean text
# --------------------------
MAX_LEN = 60          # max tokens per news title (titles are short)

def clean_text(text):
    """Simple text cleaning: lower, remove urls, emails, non-alphanum (keep spaces)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove urls
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_texts(texts, model, tokenizer, max_len=MAX_LEN):
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    seqs = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
    probs = model.predict(seqs).ravel()
    preds = (probs >= 0.5).astype(int)
    return "Real News" if preds == 1 else "Fake News", probs[0]

def print_model_info():
    
    with open('saved_model/model_metrics.txt', 'r') as file:
        s = file.readlines()
        for i in range(0, len(s)):
            s[i] = s[i].split(':')
        
    col1, col2, col3, col4, col5  = st.columns(5)
    col1.metric(s[0][0], s[0][1])
    col2.metric(s[1][0], s[1][1])
    col3.metric(s[2][0], s[2][1])
    col4.metric(s[3][0], s[3][1])
    col5.metric(s[4][0], s[4][1])


# The functions below are all used by the third tab, for the LLM based Inference of the News Data. 



            
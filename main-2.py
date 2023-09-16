import os
import pickle
import pandas as pd
import numpy as np
import random
import string
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import TextVectorization

import warnings
warnings.filterwarnings("ignore")

import re

# READING A PREPROCESSING THE DATA

with open('data/IITB.en-hi.en', 'r', encoding = 'utf-8') as f:
    english = f.readlines()

with open('data/IITB.en-hi.hi', 'r', encoding = 'utf-8') as f:
    hindi = f.readlines()

df = pd.DataFrame({'english_sent':english, 'hindi_sent':hindi})

print(df.head())

### Tranining Hyperparameters
batch_size = 256

### Model Hyperparameters
embed_dim = 128
num_heads = 10
latent_dim = 2048
vocab_size = 20000
sequence_length = 20
dropout = 0.2

# DEFINING FUNCTIONS

def preprocess_text(df):
    # Lowercase the characters
    df["english_sent"] = df["english_sent"].apply(lambda x : x.lower())
    df["hindi_sent"] = df["hindi_sent"].apply(lambda x : x.lower())

    # Rmoving URLs
    df["english_sent"] = df["english_sent"].apply(lambda x : re.sub(r"http\S+", "", x))
    df["hindi_sent"] = df["hindi_sent"].apply(lambda x : re.sub(r"http\S+", "", x))

    # Removing digits
    remove_digits = str.maketrans("", "",string.digits)
    df["english_sent"] = df["english_sent"].apply(lambda x : x.translate(remove_digits))
    df["hindi_sent"] = df["hindi_sent"].apply(lambda x : x.translate(remove_digits))
    df["hindi_sent"] = df["hindi_sent"].apply(lambda x : re.sub("[a-zA-z२३०८१५७९४६]", "", x))

    # Remove special characters
    special = set(string.punctuation)
    df['english_sent'] = df['english_sent'].apply(lambda x : ''.join(ch for ch in x if ch not in special))
    df['hindi_sent'] = df['hindi_sent'].apply(lambda x : ''.join(ch for ch in x if ch not in special))

    # Remove quotes
    df['english_sent'] = df['english_sent'].apply(lambda x: re.sub("'", '', x))
    df['hindi_sent'] = df['hindi_sent'].apply(lambda x: re.sub("'", '', x))

    # Remove extra spaces
    df['english_sent'] = df['english_sent'].apply(lambda x : x.strip())
    df['hindi_sent'] = df['hindi_sent'].apply(lambda x : x.strip())
    df['english_sent'] = df['english_sent'].apply(lambda x : re.sub(" +"," ",x))
    df['hindi_sent'] = df['hindi_sent'].apply(lambda x : re.sub(" +"," ",x))


    # Add [start] and [end] tags
    df["hindi_sent"] = df["hindi_sent"].apply(lambda x : "[start] " + x + " [end]")

def decode_sequence(input_sentence):
    hindi_vocab = hindi_vectorization.get_vocabulary()
    hindi_index_lookup = dict(zip(range(len(hindi_vocab)), hindi_vocab))
    max_decoded_sentence_length = 20

    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = hindi_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = hindi_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break

    return decoded_sentence[8:-5] # Removing [start] and [end] tokens

### For creating Dataset
def format_dataset(eng, hin):
    eng = eng_vectorization(eng)
    hindi = hindi_vectorization(hin)
    return ({"encoder_inputs" : eng, "decoder_inputs" : hindi[:, :-1],}, hindi[:, 1:])

def make_dataset(df):
    dataset = tf.data.Dataset.from_tensor_slices((df["english_sent"].values, df["hindi_sent"].values))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

### PREPROCESSING THE DATA

preprocess_text(df)

### DROPPING THE ROWS WITH NULLVALUES

df.drop(df[df["english_sent"] == " "].index, inplace = True)
df.drop(df[df["hindi_sent"] == "[start]  [end]"].index, inplace = True)

### Finding the Sentence Length

df["eng_sent_length"] = df["english_sent"].apply(lambda x : len(x.split(' ')))
df["hindi_sent_length"] = df["hindi_sent"].apply(lambda x : len(x.split(' ')))

### Get sentences with specific length 20
df = df[df["eng_sent_length"] <= 20]
df = df[df["hindi_sent_length"] <= 20]

# TAKING 85K RECORDS FOR THE TRAINING
df = df.sample(n = 85000, random_state = 2048)
df = df.reset_index(drop = True)

# DEFINING TRAIN, VALID, TEST
train = df.iloc[:80000]
val = df.iloc[80000:84500]
test = df.iloc[84500:]

# TOKENIZING SENTENCES

## Using TextVectorization to create sentence vectors
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(
    max_tokens = vocab_size, output_mode = "int", output_sequence_length = sequence_length
    )

hindi_vectorization = TextVectorization(
    max_tokens = vocab_size, output_mode = "int", output_sequence_length = sequence_length + 1, standardize=custom_standardization
)

eng_vectorization.adapt(df["english_sent"].values)
hindi_vectorization.adapt(df["hindi_sent"].values)

### Savng parameters and weights of both vectorizer
pickle.dump({'config': eng_vectorization.get_config(),
             'weights': eng_vectorization.get_weights()}
            , open("eng_vectorizer.pkl", "wb"))

pickle.dump({'config': hindi_vectorization.get_config(),
             'weights': hindi_vectorization.get_weights()}
            , open("hindi_vectorizer.pkl", "wb"))

# CREATING DATASET

train_ds = make_dataset(train)
val_ds = make_dataset(val)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_len, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = layers.Embedding(
            input_dim = vocab_size, output_dim = embed_dim
        )
        self.position_embedding = layers.Embedding(
            input_dim = sequence_len, output_dim = embed_dim
        )

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start = 0, limit = length, delta = 1)
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, dropout,**kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim
        )
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.layer_ffn = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"),
             layers.Dropout(dropout),
             layers.Dense(embed_dim),]
            )
        self.supports_masking = True

    def call(self, inputs, mask = None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")

        attention_output = self.attention(
            query = inputs, value = inputs, key = inputs, attention_mask = padding_mask
        )
        ffn_input = self.layer_norm1(inputs + attention_output)
        ffn_output = self.layer_ffn(ffn_input)
        return self.layer_norm2(ffn_input + ffn_output)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, sropout,**kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention1 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim
        )
        self.attention2 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim
        )
        self.layer_ffn = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"),
             layers.Dropout(dropout),
             layers.Dense(embed_dim),]
        )
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.layer_norm3 = layers.LayerNormalization()

        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask = None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output1 = self.attention1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out1 = self.layer_norm1(inputs + attention_output1)

        attention_output2 = self.attention2(
            query = out1, value = encoder_outputs, key = encoder_outputs, attention_mask = padding_mask
        )
        out2 = self.layer_norm2(out1 + attention_output2)

        ffn_output = self.layer_ffn(out2)
        return self.layer_norm3(out2 + ffn_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads, dropout,name="encoder_1")(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads, dropout,name="decoder_1")(x, encoded_seq_inputs)
x = layers.Dropout(0.4)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

transformer.summary()

# TRAINING MODEL

### Defining callback functions
early_stopping = EarlyStopping(patience = 5,restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

### Compiling model
transformer.compile(
    optimizer = "adam",
    loss="sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

### Training model
transformer.fit(train_ds, epochs = 10, validation_data = val_ds, callbacks = [early_stopping, reduce_lr])

### Saving weights of model
transformer.save_weights("eng-hin.h5")

# TESTING THE MODEL & CALCULATING BLEU SCORE

### Sample for testing
eng = "how are you"
print("English Sentence : ",eng)
print("Translated Sentence : ",decode_sequence(eng))

### Calculating BLEU Score

eng = test["english_sent"].values
original = test["hindi_sent"].values

translated = [decode_sequence(sent) for sent in eng]
bleu = 0

for i in range(test.shape[0]):
    bleu += sentence_bleu([original[i].split()], translated[i].split(), weights = (0.5, 0.5))

print("BLEU score is : ", bleu / test.shape[0])
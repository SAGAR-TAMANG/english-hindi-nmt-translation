import string
import re

def preprocess(text):
  text = ''.join(ch for ch in text if ch not in string.punctuation)
  text = text.lower()
  text = re.sub(r'\d','',text)
  text = re.sub(r'\s+',' ',text)  #Remove extra spaces
  text = text.strip()
  return text

total_sentences = 25000    #Total sentences
maxlen = 10    #Max length of sentences for both English and Hindi

en_data = []
hi_data = []

cnt = 0

hindi_sentences = re.sub(r'[a-zA-Z]','',hi) 

for hi in hindi_sentences

for (en,hi) in zip(english_sentences, hindi_sentences):
  l = min(len(en.split()), len(hi.split()))
  if l <= maxlen:
    en_data.append(en)
    hi_data.append(hi)
    cnt += 1
  if cnt == total_sentences:
    break

#Tokenize the texts and convert to sequences
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>', lower=False)
en_tokenizer.fit_on_texts(en_data)
en_sequences = en_tokenizer.texts_to_sequences(en_data)

hi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>', lower=False)
hi_tokenizer.fit_on_texts(hi_data)
hi_sequences = hi_tokenizer.texts_to_sequences(hi_data)

#To incorporate padding zeros
english_vocab_size = len(en_tokenizer.word_index) + 1
hindi_vocab_size = len(hi_tokenizer.word_index) + 1
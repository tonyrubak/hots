from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, Activation, LSTM, Dropout
import pandas as pd
import numpy as np

max_features = 100

# Build the vocabulary
vocab = pd.read_csv("dictionary.csv", header=None)[0]
vocab_dict = dict()
idx_dict = dict()
next_words = []

for (word_idx, word) in enumerate(vocab):
    vocab_dict[word] = word_idx
    idx_dict[word_idx] = word

# Load the training data
drafts = pd.read_csv("drafts.csv")
for i in range(0, len(drafts)):
    for j in range(0, len(drafts.iloc[i]) - 1):
        next_words.append(drafts.iloc[i,j+1])

train_x = np.zeros((len(drafts), 17, len(vocab)), dtype=np.bool)
train_y = np.zeros((len(drafts), len(vocab)), dtype=np.bool)
for (i, draft) in enumerate(drafts.iterrows()):
    for (t, word) in enumerate(drafts.iloc[i]):
        train_x[i, t, vocab_dict[word]] = 1
    train_y[i, vocab_dict[next_words[i]]] = 1

model = Sequential()
model.add(SimpleRNN(32, return_sequences=False, input_shape=(17, len(vocab))))
model.add(Dense(len(vocab), activation="softmax"))

model = Sequential()
model.add(LSTM(32, return_sequences=False, input_shape=(17, len(vocab))))
model.add(Dropout(0.2))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
              metrics=["acc"])


model.fit(train_x, train_y, batch_size = 2, epochs = 10)
start_idx = np.random.randint(83, 97)

print(idx_dict[start_idx])
generated = idx_dict[start_idx] + " "
next_word = idx_dict[start_idx]

for i in range(16):
    x = np.zeros((1, 17, len(vocab)))
    x[0, 0, vocab_dict[next_word]] = 1.
    
    preds = model.predict(x)[0]
    next_idx = np.random.multinomial(1, preds)
    next_idx = np.argwhere(next_idx == 1)[0,0]
    next_word = idx_dict[next_idx]
    generated += next_word + " "
    print(next_word)

print(generated)

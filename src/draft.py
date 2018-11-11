from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, SimpleRNN, Activation, LSTM, Dropout, Input, TimeDistributed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
for (i, draft) in enumerate(drafts.sample(frac=1).iterrows()):
    for (t, word) in enumerate(drafts.iloc[i]):
        train_x[i, t, vocab_dict[word]] = 1
    train_y[i, vocab_dict[next_words[i]]] = 1

def fit_new_model(train_x, train_y):
    train_y = train_y.reshape(-1,1,98)
    train_y = np.repeat(train_y, 17, axis=1)
    model = Sequential()
    model.add(SimpleRNN(17, return_sequences = True, input_shape=(17, len(vocab))))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(98, kernel_regularizer=regularizers.l2(0.01))))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["acc"])
    history = model.fit(train_x, train_y, batch_size = 100, epochs = 50, validation_split=0.2)
    return (history, model)

def generate_draft(model):
    start_idx = np.random.randint(84, 96)
    while start_idx == 85 or start_idx == 89 or start_idx == 90 or start_idx == 91:
        start_idx = np.random.randint(84, 96)    
    generated = idx_dict[start_idx] + " "
    next_word = idx_dict[start_idx]
    x = np.zeros((1, 17, len(vocab)))
    x[0, 0, vocab_dict[next_word]] = 1.
    preds = model.predict(x)[0]
    res = []
    for (i, pred) in enumerate(preds):
        next_idx = 100
        while (next_idx > 83) | (next_idx in res[:i]):
            next_idx = np.argwhere(np.random.multinomial(1, pred) == 1)[0,0]
        res.append(next_idx)
    # pmap = map(lambda x: np.random.multinomial(1,x), preds)
    # words = [idx_dict[np.argwhere(p == 1)[0,0]] for p in res]
    words = [idx_dict[r] for r in res]
    return (idx_dict[start_idx], words)

def plot_model(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training accuarcy")
    plt.legend()
    # ax = plt.gca()
    # ax.set_yscale("log")
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training loss")
    plt.legend()
    # ax = plt.gca()
    # ax.set_yscale("log")
    plt.show()

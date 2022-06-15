import random
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau

my_path = "data" + "\\" + "tng_scripts"
script_files = [join(my_path, f) for f in listdir(my_path) if isfile(join(my_path, f))]

all_text = ""
scripts = []
for script_file in script_files:
    print(script_file)
    with open(script_file, 'r') as file:
        text = file.read()
        scripts.append(text)

print(len(scripts))
#print(("".join(scripts)))

chars = sorted(list(set("".join(scripts))))
print("total unique characters: ", len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Break up the scripts into "sentences"
maxlen = 40
step = 3
sentences = []
next_chars = []
for script in scripts:
    for i in range(0, len(script) - maxlen, step):
        sentences.append(script[i: i + maxlen])
        next_chars.append(script[i + maxlen])

print("total sentences:", len(sentences))

print("Initializing inputs...")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print("Input initialization complete.")

# Create the model
print("Initializing model...")
model = Sequential()
model.add(LSTM(1024, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = Adam(lr=0.0001)
# model.load_weights('tng_init_weights.hdf5')
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print("Model initialization complete.")

# serialize model to JSON
model_json = model.to_json()
with open("textgen_model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
filepath = "tng_weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.00003)
callbacks = [print_callback, checkpoint, reduce_lr]

print("Training model...")
model.fit(x, y, batch_size=128, epochs=200, callbacks=callbacks)
print("Model training complete")

# TODO Try two LSTM layers (with dropout on each) and one Dense layer

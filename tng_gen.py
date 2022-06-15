from datetime import datetime
import sys
from os import listdir
from os.path import join, isfile

import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.models import model_from_json
from keras.optimizers import Adam

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

maxlen = 40


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_script(model, temperature, start_text, length):
    # Function invoked at end of each epoch. Prints generated text.

    f = open("tng_script_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".txt", "w")

    generated = ''
    sentence = start_text
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    f.write(generated)

    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        f.write(next_char)
        sys.stdout.flush()
    print()
    f.flush()
    f.close()


# load json and create model
print("Initializing model...")
json_file = open('textgen_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("tng_init_weights.hdf5")
print("Loaded model from disk")
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print("Model initialization complete.")
# Create the model

# print("Initializing model...")
# model = Sequential()
# model.add(LSTM(1024, input_shape=(maxlen, len(chars)), return_sequences=True))
# model.add(LSTM(1024, input_shape=(maxlen, len(chars))))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))
# optimizer = Adam(lr=0.0001)
# model.load_weights('tng_init_weights.hdf5')
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# print("Model initialization complete.")

generate_script(model, 0.75, "\"That's My Q\"".rjust(40, " "), 70000)
print("Script complete!")

DATA_PATH = 'warpeace.txt'

data = open(DATA_PATH, 'r').read()
chars = list(set(data)) #set: gets unique values
VOCAB_SIZE = len(chars)
len(data)

print('chars:\n{}\n\nVOCAB_SIZE: {}'.format(chars, VOCAB_SIZE))

idx_to_char = {i: char for i, char in enumerate(chars)}
char_to_idx = {char: i for i, char in enumerate(chars)}

import numpy as np

SEQ_LENGTH = 60 #input sequence length
N_FEATURES = VOCAB_SIZE #one hot encoding here, that's why, but deduplicated for clarity

N_SEQ = int(np.floor((len(data) - 1) / SEQ_LENGTH))

X = np.zeros((N_SEQ, SEQ_LENGTH, N_FEATURES))
y = np.zeros((N_SEQ, SEQ_LENGTH, N_FEATURES))

for i in range(N_SEQ):
  X_sequence = data[i * SEQ_LENGTH: (i + 1) * SEQ_LENGTH]
  X_sequence_ix = [char_to_idx[c] for c in X_sequence]
  input_sequence = np.zeros((SEQ_LENGTH, N_FEATURES))
  for j in range(SEQ_LENGTH):
    input_sequence[j][X_sequence_ix[j]] = 1. #one-hot encoding of the input characters
  X[i] = input_sequence
  
  y_sequence = data[i * SEQ_LENGTH + 1: (i + 1) * SEQ_LENGTH + 1] #shifted by 1 to the right
  y_sequence_ix = [char_to_idx[c] for c in y_sequence]
  target_sequence = np.zeros((SEQ_LENGTH, N_FEATURES))
  for j in range(SEQ_LENGTH):
    target_sequence[j][y_sequence_ix[j]] = 1. #one-hot encoding of the target characters
  y[i] = target_sequence
  
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation

# constant parameter for the model
HIDDEN_DIM = 700 #size of each hidden layer, "each layer has 700 hidden states"
LAYER_NUM = 2 #number of hidden layers, how much were used?
NB_EPOCHS = 200 #max number of epochs to train, "200 epochs"
BATCH_SIZE = 128 
VALIDATION_SPLIT = 0.1 #proportion of the batch used for validation at each epoch


model = Sequential()
model.add(LSTM(HIDDEN_DIM, 
               input_shape=(None, VOCAB_SIZE), 
               return_sequences=True))
for _ in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])


def generate_text(model, length):
  ix = [np.random.randint(VOCAB_SIZE)]
  y_char = [idx_to_char[ix[-1]]]
  X = np.zeros((1, length, VOCAB_SIZE))
  for i in range(length):
    X[0, i, :][ix[-1]] = 1.
    ix = np.argmax(model.predict(X[:, :i+1,:])[0], 1)
    y_char.append(idx_to_char[ix[-1]])
  return ''.join(y_char)

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
# callback to save the model if better
filepath="tgt_model.hdf5"
save_model_cb = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callback to stop the training if no improvement
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=0)
# callback to generate text at epoch end
class generateText(Callback):
    def on_epoch_end(self, batch, logs={}):
        print(generate_text(self.model, 100))
        
generate_text_cb = generateText()

callbacks_list = [save_model_cb, early_stopping_cb, generate_text_cb]

model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=NB_EPOCHS, callbacks=callbacks_list, validation_split=VALIDATION_SPLIT)
















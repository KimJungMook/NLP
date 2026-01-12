import tensorflow as tf
import os
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from keras import Input, Model, Optimizer
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

# print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in X_train)))
# print('리뷰의 평균 길이 : {}'.format(sum(map(len, X_train))/len(X_train)))

max_len = 500
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)



class BahdanauAttention(tf.keras.Model):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(
            tf.nn.tanh(
                self.W1(values) + self.W2(hidden_with_time_axis)
            )
        )

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

sequence_input = Input(shape= (max_len, ), dtype = 'int32')
embedded_sequences = (Embedding(vocab_size, 128, input_length = max_len,
                                mask_zero = True)(sequence_input))

lstm = Bidirectional(LSTM(64, dropout= 0.5, return_sequences = True))(embedded_sequences)

lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
    (LSTM(64, dropout = 0.5, return_sequences = True, return_state = True))(lstm)

# print(lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)

state_h = Concatenate()([forward_h, backward_h]) # 은닉상태
state_c = Concatenate()([forward_c, backward_c])

attention = BahdanauAttention(64)
context_vector, attention_weights = attention(lstm, state_h)

dense1 = Dense(20, activation = 'relu')(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation = 'sigmoid')(dropout)
model = Model(inputs = sequence_input, outputs = output)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# history = model.fit(X_train, y_train, epochs = 3, batch_size = 256,
#                     validation_data = (X_test, y_test), verbose = 1)

print(model.evaluate(X_test, y_test)[1])
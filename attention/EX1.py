from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

# print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in X_train)))
# print('리뷰의 평균 길이 : {}'.format(sum(map(len, X_train))/len(X_train)))

max_len = 500
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)


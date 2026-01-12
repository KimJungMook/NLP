import tensorflow as tf
import numpy as np

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name = "multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units = d_model)
        self.key_dense = tf.keras.layers.Dense(units = d_model)
        self.value_dense = tf.keras.layers.Dense(units = d_model)

        self.dense = tf.keras.layers.Dense(units = d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape = (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(inputs, perm = [0, 2, 1, 3])

    def call(self, inputs):
        query, key , value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, key의 문장 길이)
  return mask[:, tf.newaxis, tf.newaxis, :]


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b = True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis = -1)

    output = tf.matmul(attention_weights, value)

    return output, attention_weights


np.set_printoptions(suppress=True)

temp_k = tf.constant([
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10],
    [0, 0, 10]], dtype = tf.float32)

temp_v = tf.constant([
    [1, 0],
    [10, 0],
    [100, 5],
    [1000, 6]], dtype = tf.float32)

temp_q = tf.constant([[0, 10, 0]], dtype = tf.float32)

temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)

# print(temp_out)
# print(temp_attn)

temp_q = tf.constant([[0, 0, 10]], dtype = tf.float32)
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)

# print(temp_out)
# print(temp_attn)


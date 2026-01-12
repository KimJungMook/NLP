import tensorflow as tf

from transformer.EX1 import PositionalEncoding
from transformer.EX2 import MultiHeadAttention, create_padding_mask


# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
  return tf.maximum(look_ahead_mask, padding_mask)



def encoder_layer(dff, d_model, num_heads, dropout, name = "encoder_layer"):
    inputs = tf.keras.Input(shape = (None, d_model), name = "inputs")

    padding_mask = tf.keras.Input(shape = (1, 1, None), name = "padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name = "attention")({
        'query' : inputs, 'key' : inputs, 'value' : inputs,
        'mask' : padding_mask
    })

    attention = tf.keras.layers.Dropout(rate = dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon = 1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units = dff, activation = "relu")(attention)
    outputs = tf.keras.layers.Dense(units = d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate = dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon = 1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs = [inputs, padding_mask], outputs = outputs, name = name)


def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name = "encoder"):
    inputs = tf.keras.Input(shape = (None, ), name = "inputs")

    padding_mask = tf.keras.Input(shape = (1, 1, None), name = "padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32)
    )(embeddings)
    embeddings = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32) * tf.math.sqrt(tf.cast(d_model, tf.float32))
    )(embeddings)
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate = dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(dff = dff, d_model = d_model, num_heads = num_heads,
                                dropout = dropout, name = "encoder_layer_{}".format(i))([outputs, padding_mask])

    return tf.keras.Model(
        inputs = [inputs, padding_mask], outputs = outputs, name = name)


def decoder_layer(dff, d_model, num_heads, dropout, name = "decoder_layer"):
    inputs = tf.keras.Input(shape = (None, d_model), name = "inputs")

    enc_outputs = tf.keras.Input(shape = (None, d_model), name = "encoder_outputs")

    look_ahead_mask = tf.keras.Input(
        shape = (1, None, None), name = "look_ahead_mask")

    padding_mask = tf.keras.Input(shape = (1, 1, None), name = "padding_mask")

    attention1 = MultiHeadAttention(
        d_model, num_heads, name = "attention1")(inputs ={
        'query' : inputs, 'key' : inputs, 'value' : inputs,
        'mask' : look_ahead_mask
    })

    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name = "attention2")(inputs = {
        'query' : attention1, 'key' : enc_outputs, 'value' : enc_outputs,
        'mask' : padding_mask
    })

    attention2 = tf.keras.layers.Dropout(rate = dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units = dff, activation = "relu")(attention2)
    outputs = tf.keras.layers.Dense(units = d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate = dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs = [inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs = outputs, name = name
    )


def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name = "decoder"):

    inputs = tf.keras.Input(shape = (None, ), name = "inputs")
    enc_outputs = tf.keras.Input(shape = (None, d_model), name = "encoder_outputs")

    look_ahead_mask = tf.keras.Input(
        shape = (1,None, None), name = 'look_ahead_mask')
    padding_mask = tf.keras.Input(shape = (1, 1, None), name = 'padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32)
    )(embeddings)
    embeddings = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32) * tf.math.sqrt(tf.cast(d_model, tf.float32))
    )(embeddings)
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate = dropout)(embeddings)

    for i in range(num_layers):
        outputs = (decoder_layer(dff = dff, d_model = d_model, num_heads = num_heads,
                                dropout = dropout, name = "decoder_layer_{}".format(i))
                   ([outputs, enc_outputs, look_ahead_mask, padding_mask]))

    return tf.keras.Model(
        inputs = [inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs = outputs,
        name = name
    )

def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name = "transformer"):

    inputs = tf.keras.Input(shape = (None, ), name = "inputs")

    dec_inputs = tf.keras.Input(shape = (None, ), name = "dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape = (1, 1, None),
        name = "enc_padding_mask"
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape = (1, None, None),
        name = "look_ahead_mask")(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape = (1, 1, None),
        name = "dec_padding_mask")(inputs)

    enc_outputs = encoder(vocab_size = vocab_size, num_layers = num_layers, dff = dff,
                          d_model = d_model, num_heads = num_heads, dropout = dropout)(inputs = [inputs, enc_padding_mask])

    dec_outputs = decoder(vocab_size = vocab_size, num_layers = num_layers, dff = dff,
                          d_model = d_model, num_heads = num_heads, dropout = dropout)(inputs = [dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units = vocab_size, name = "outputs")(dec_outputs)

    return tf.keras.Model(inputs = [inputs, dec_inputs], outputs = outputs, name = name)


small_transformer = transformer(
    vocab_size = 9000,
    num_layers = 4,
    dff = 512,
    d_model = 128,
    num_heads = 4,
    dropout = 0.3,
    name = "small_transformer")

# tf.keras.utils.plot_model(
#     small_transformer, to_file = 'small_transformer.png', show_shapes = True)

print(small_transformer.summary())

# 또는 더 자세한 정보
for layer in small_transformer.layers:
    print(f"{layer.name}: {layer}")

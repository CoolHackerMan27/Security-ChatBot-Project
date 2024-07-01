from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding
import numpy as np
import tensorflow as tf
import tensorflow_text as text


tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Load data in chunks


def load_data_chunk(filename, chunk_size=10000):
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.readlines(chunk_size)
            if not chunk:
                break
            yield chunk


# Create a tokenizer
tokenizer = text.BertTokenizer('bert_vocab.txt', lower_case=True)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float16)

# Define constants
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DFF = 1024
PE_INPUT = 1000
PE_TARGET = 1000
EPOCHS = 10


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float16)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            Dense(d_model)  # (batch_size, seq_len, d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, d_model, dtype=tf.float16)
        self.pos_encoding = tf.cast(positional_encoding(maximum_position_encoding, d_model), dtype=tf.float16)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training=False, mask=None):
        seq_len = tf.shape(x)[1]
        
        # Cast input to int32 (if it's not already)
        x = tf.cast(x, tf.int32)
        
        # Embed and cast to float16
        x = self.embedding(x)
        x = tf.cast(x, tf.float16)
        
        # Scale
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float16))
        
        # Add positional encoding
        x += tf.cast(self.pos_encoding[:, :seq_len, :], tf.float16)
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x
class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            Dense(d_model)  # (batch_size, seq_len, d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x = tf.cast(x, tf.float16)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float16))
        x += tf.cast(self.pos_encoding[:, :seq_len, :], tf.float16)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        enc_output = self.encoder(inputs, training=training, mask=enc_padding_mask)
        dec_output, attention_weights = self.decoder(targets, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


# Gradient accumulation steps
ACCUMULATION_STEPS = 4


@tf.function
def train_step(inp, tar):
    inp = tf.cast(inp, tf.int32)
    tar = tf.cast(tar, tf.int32)
    
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask = tf.cast(create_padding_mask(inp), tf.float16)
    look_ahead_mask = tf.cast(create_look_ahead_mask(tf.shape(tar_inp)[1]), tf.float16)
    dec_padding_mask = tf.cast(create_padding_mask(inp), tf.float16)

    with tf.GradientTape() as tape:
        predictions, _ = model(
            inputs=inp, 
            targets=tar_inp, 
            training=True, 
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=look_ahead_mask,
            dec_padding_mask=dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)
        loss = loss / ACCUMULATION_STEPS

    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients
# Use mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.AdamW(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
)

BUFFER_SIZE = 10000
BATCH_SIZE = 32

# Initialize the model
model = TransformerModel(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=2050,
    target_vocab_size=2050,
    pe_input=PE_INPUT,
    pe_target=PE_TARGET
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def pad_sequences_to_same_length(seq1, seq2, padding='post'):
    # Find the maximum length between the two sequences
    max_length = max(tf.shape(seq1)[1], tf.shape(seq2)[1])
    # Pad the sequences to the same length
    padded_seq1 = tf.pad(seq1, [[0, 0], [0, max_length - tf.shape(seq1)[1]]], constant_values=0)
    padded_seq2 = tf.pad(seq2, [[0, 0], [0, max_length - tf.shape(seq2)[1]]], constant_values=0)
    return padded_seq1, padded_seq2

def tokenize_and_encode(chunk):
    tokenized_text = tokenizer.tokenize(chunk)
    return tokenized_text.merge_dims(-2, -1)

# In the training loop:
for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

    for input_chunk, target_chunk in zip(load_data_chunk('train.from'), load_data_chunk('train.to')):
        train_inputs_encoded = tokenize_and_encode(input_chunk)
        train_targets_encoded = tokenize_and_encode(target_chunk)

        # Convert to dense tensors
        train_inputs_encoded = train_inputs_encoded.to_tensor()
        train_targets_encoded = train_targets_encoded.to_tensor()

        # Ensure same number of samples
        min_samples = min(train_inputs_encoded.shape[0], train_targets_encoded.shape[0])
        train_inputs_encoded = train_inputs_encoded[:min_samples]
        train_targets_encoded = train_targets_encoded[:min_samples]

        # Pad sequences to the same length
        train_inputs_encoded, train_targets_encoded = pad_sequences_to_same_length(
            train_inputs_encoded, train_targets_encoded)

        # Debug: Print the shapes after padding
        print(f"After padding - Inputs shape: {train_inputs_encoded.shape}, Targets shape: {train_targets_encoded.shape}")

        dataset = tf.data.Dataset.from_tensor_slices((train_inputs_encoded, train_targets_encoded))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        for (batch, (inp, tar)) in enumerate(dataset):
            batch_loss, gradients = train_step(inp, tar)
            total_loss += batch_loss
            batch_count += 1

            accumulated_gradients = [accu_grad + grad for accu_grad, grad in zip(accumulated_gradients, gradients)]

            if batch_count % ACCUMULATION_STEPS == 0:
                ptimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

    print(f"Epoch {epoch + 1} Loss: {total_loss/batch_count}")
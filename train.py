import tensorflow as tf
from model import TransformerModel
from data_loader import DataGenerator, pad_sequences_to_same_length
from utils import CustomSchedule, loss_function
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DFF = 1024
PE_INPUT = 1000
PE_TARGET = 1000
EPOCHS = 10
BUFFER_SIZE = 10000
BATCH_SIZE = 32
ACCUMULATION_STEPS = 4

# Initialize tokenizer
tokenizer = text.BertTokenizer('bert_vocab.txt', lower_case=True)

# Initialize the model
model = TransformerModel(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=205000,
    target_vocab_size=205000,
    pe_input=PE_INPUT,
    pe_target=PE_TARGET
)

# Set up optimizer and loss
learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.AdamW(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# Set up data generator
train_generator = DataGenerator('train.from', 'train.to', BATCH_SIZE, tokenizer)

@tf.function
def train_step(inp, tar):
    inp = tf.cast(inp, tf.int32)
    tar = tf.cast(tar, tf.int32)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask = tf.cast(create_padding_mask(inp), tf.float16)
    look_ahead_mask = tf.cast(create_look_ahead_mask(
        tf.shape(tar_inp)[1]), tf.float16)
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
        predictions = tf.debugging.check_numerics(predictions, "NaN in model output")
        loss = loss_function(tar_real, predictions)
        loss = loss / ACCUMULATION_STEPS

    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.cast(grad, tf.float32) for grad in gradients]
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    if any(tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients):
        tf.print("NaN detected in gradients")
        return loss, None
    
    return loss, gradients
# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

    for inp, tar in train_generator:
        inp, tar = pad_sequences_to_same_length(inp, tar)
        
        try:
            batch_loss, gradients = train_step(inp, tar)
            if gradients is not None:
                total_loss += batch_loss
                batch_count += 1

                accumulated_gradients = [accu_grad + grad for accu_grad, grad in zip(accumulated_gradients, gradients)]

                if batch_count % ACCUMULATION_STEPS == 0:
                    optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
        except tf.errors.ResourceExhaustedError:
            logging.warning("Out of memory. Skipping this batch.")
            continue

    logging.info(f"Epoch {epoch + 1} Loss: {total_loss/batch_count if batch_count > 0 else 0}")

    # Add checkpointing and validation here if needed
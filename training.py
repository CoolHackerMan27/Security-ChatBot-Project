import tensorflow as tf
import tensorflow_text as text

# Load your data in chunks


def load_data_chunk(filename, chunk_size=10000):
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.readlines(chunk_size)
            if not chunk:
                break
            yield chunk


# Create a tokenizer
tokenizer = text.BertTokenizer('bert_vocab.txt', lower_case=True)

# Define constants (reduced sizes)
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DFF = 1024
PE_INPUT = 1000
PE_TARGET = 1000

# Transformer model class (unchanged)


class TransformerModel(tf.keras.Model):
    # ... (keep the same implementation)

    # Tokenize and encode the data


def tokenize_and_encode(texts):
    return tokenizer.tokenize(texts).to_tensor()

# Custom learning rate scheduler (unchanged)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # ... (keep the same implementation)

    # Loss function (unchanged)


def loss_function(real, pred):
    # ... (keep the same implementation)

    # Gradient accumulation steps
ACCUMULATION_STEPS = 4


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp, True, None, None, None)
        loss = loss_function(tar_real, predictions)
        loss = loss / ACCUMULATION_STEPS

    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients


# Use mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

BUFFER_SIZE = 10000
BATCH_SIZE = 32

# Initialize the model
model = TransformerModel(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=tokenizer.vocab_size(),
    target_vocab_size=tokenizer.vocab_size(),
    pe_input=PE_INPUT,
    pe_target=PE_TARGET
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    accumulated_gradients = [tf.zeros_like(
        var) for var in model.trainable_variables]

    for input_chunk in load_data_chunk('train.from'):
        target_chunk = next(load_data_chunk('train.to'))

        train_inputs_encoded = tokenize_and_encode(input_chunk)
        train_targets_encoded = tokenize_and_encode(target_chunk)

        dataset = tf.data.Dataset.from_tensor_slices(
            (train_inputs_encoded, train_targets_encoded))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(
            BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        for (batch, (inp, tar)) in enumerate(dataset):
            batch_loss, gradients = train_step(inp, tar)
            total_loss += batch_loss
            batch_count += 1

            accumulated_gradients = [
                accu_grad + grad for accu_grad, grad in zip(accumulated_gradients, gradients)]

            if batch_count % ACCUMULATION_STEPS == 0:
                optimizer.apply_gradients(
                    zip(accumulated_gradients, model.trainable_variables))
                accumulated_gradients = [tf.zeros_like(
                    var) for var in model.trainable_variables]

    print(f'Epoch {epoch + 1}, Loss: {total_loss/batch_count}')

# Save the model
model.save('RedditSLMv1.0.h5')

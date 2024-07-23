import tensorflow as tf
from model import TransformerModel
from dataLoader import get_data_pipeline
from utils import CustomSchedule, loss_function, create_padding_mask, create_look_ahead_mask
import logging
from tensorflow.keras import mixed_precision

# Enable mixed precision training for faster compute on modern GPU
mixed_precision.set_global_policy('mixed_float16')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DFF = 1024
PE_INPUT = 1000
PE_TARGET = 1000
EPOCHS = 10
BATCH_SIZE = 256
ACCUMULATION_STEPS = 4
INPUT_FILE = 'train.from'
TARGET_FILE = 'train.to'
MAX_LENGTH = 512
WARMUP_STEPS = 4000

# Get Tokenizer and Data Generator
tokenizer, dataset = get_data_pipeline(INPUT_FILE, TARGET_FILE, BATCH_SIZE, MAX_LENGTH)

# Initialize the model
model = TransformerModel(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=len(tokenizer),
    target_vocab_size=len(tokenizer),
    pe_input=PE_INPUT,
    pe_target=PE_TARGET
)

# Set up optimizer and loss
learning_rate = CustomSchedule(D_MODEL, warmup_steps=WARMUP_STEPS)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask = create_padding_mask(inp)
    combined_mask = tf.maximum(
        create_padding_mask(tar_inp),
        create_look_ahead_mask(tf.shape(tar_inp)[1])
    )
    dec_padding_mask = create_padding_mask(inp)
    
    with tf.GradientTape() as tape:
        predictions, _ = model(
            inputs=inp,
            targets=tar_inp,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    #cast all gradients to float32
    gradients = [tf.cast(g, tf.float32) for g in gradients]
    #clip gradients to prevent exploding gradients (boom!)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.5)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
#Testing
    for item in dataset.take(1): print(item)

    for batch in dataset:        
            input_ids = batch['input_ids']
            labels = batch['labels']
            batch_loss = train_step(input_ids, labels)
            total_loss += batch_loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                logging.info(f'Epoch {epoch + 1} Batch {num_batches} Loss {batch_loss.numpy():.4f}')
    
    avg_loss = total_loss / num_batches
    logging.info(f'Epoch {epoch + 1} Loss {avg_loss:.4f}')



# Save the model
model.save_weights('transformer_model_weights')
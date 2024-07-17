from transformers import TFAutoModelForCausalLM
from dataLoader import get_data_pipeline
import tensorflow as tf

# Define constants
BATCH_SIZE = 32
MAX_LENGTH = 512
INPUT_FILE = 'train.from'
TARGET_FILE = 'train.to'

# Get the tokenizer and data generator
tokenizer, train_generator = get_data_pipeline(INPUT_FILE, TARGET_FILE, BATCH_SIZE, MAX_LENGTH)

# Initialize the model
model = TFAutoModelForCausalLM.from_pretrained("gpt2")

# Compile the model
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)
model.compile(optimizer='AdamW', loss='sparse_categorical_crossentropy')

# Train the model (This is broken right now, but you get the idea)
#model.fit(train_generator, epochs=10)

# Function to generate responses
def generate_response(instruction, max_length=100):
    input_text = f"Instruction: {instruction}\nResponse:"
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

while(True):
    instruction = intput("Enter instruction: ")
    response = generate_response(instruction)
    print(response)
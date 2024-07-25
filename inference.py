import tensorflow as tf
from model import TransformerModel
from utils import create_padding_mask, create_look_ahead_mask
from dataLoader import get_tokenizer

# Define constants (make sure these match your training setup)
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DFF = 1024
PE_INPUT = 1000
PE_TARGET = 1000
MAX_LENGTH = 512

def load_model(checkpoint_path):
    # Initialize the model
    tokenizer = get_tokenizer()
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
    
    # Load the weights
    model.load_weights(checkpoint_path)
    return model, tokenizer

def preprocess_input(input_text, tokenizer):
    # Tokenize and pad the input
    input_tokens = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="tf")
    input_tokens = input_tokens[:, :MAX_LENGTH]  # Truncate if too long
    input_tokens = tf.pad(input_tokens, [[0, 0], [0, MAX_LENGTH - tf.shape(input_tokens)[1]]])
    return input_tokens

def inference(model, tokenizer, input_text):
    # Preprocess input
    input_tensor = preprocess_input(input_text, tokenizer)
    
    # Create masks
    enc_padding_mask = create_padding_mask(input_tensor)
    combined_mask = tf.maximum(
        create_padding_mask(tf.zeros_like(input_tensor)),
        create_look_ahead_mask(tf.shape(input_tensor)[1])
    )
    dec_padding_mask = create_padding_mask(input_tensor)
    
    # Initialize target input with start token (for GPT-2, we can use the same input)
    output = input_tensor
    
    for i in range(MAX_LENGTH):
        predictions, _ = model(
            inputs=input_tensor,
            targets=output,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )
        
        # Get the last token prediction
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # Break if end token is predicted (for GPT-2, you might want to use a specific end token or condition)
        if predicted_id == tokenizer.eos_token_id:
            break
        
        # Concatenate the predicted token to the output
        output = tf.concat([output, predicted_id], axis=-1)
    
    # Decode the output tokens
    return tokenizer.decode(tf.squeeze(output).numpy())

# Main execution
if __name__ == "__main__":
    checkpoint_path = 'transformer_model.weights.h5'
    model, tokenizer = load_model(checkpoint_path)
    while True:
        input_text = input("Enter input: ")
        response = inference(model, tokenizer, input_text)
        print(response)
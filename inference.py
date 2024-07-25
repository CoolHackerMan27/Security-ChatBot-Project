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
    
    model.load_weights(checkpoint_path)
    return model, tokenizer

def preprocess_input(input_text, tokenizer):
    input_tokens = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="tf")
    input_tokens = input_tokens[:, :MAX_LENGTH]  # Truncate if too long
    input_length = tf.shape(input_tokens)[1]
    padding_length = MAX_LENGTH - input_length
    input_tokens = tf.pad(input_tokens, [[0, 0], [0, padding_length]])
    return input_tokens, input_length

def create_masks(input_tensor, target_tensor, input_length):
    enc_padding_mask = create_padding_mask(input_tensor)
    dec_padding_mask = create_padding_mask(input_tensor)
    
    look_ahead_mask = create_look_ahead_mask(MAX_LENGTH)
    dec_target_padding_mask = create_padding_mask(target_tensor)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    # Adjust masks to account for the actual input length
    combined_mask = combined_mask[:, :, :input_length, :input_length]
    
    return enc_padding_mask, combined_mask, dec_padding_mask

def inference(model, tokenizer, input_text):
    input_tensor, input_length = preprocess_input(input_text, tokenizer)
    
    output = input_tensor
    
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_tensor, output, input_length)
        
        predictions, _ = model(
            inputs=input_tensor,
            targets=output,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )
        
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        if predicted_id == tokenizer.eos_token_id:
            break
        
        output = tf.concat([output, predicted_id], axis=-1)
        input_length += 1
    
    return tokenizer.decode(tf.squeeze(output).numpy())

# Main execution
if __name__ == "__main__":
    checkpoint_path = 'transformer_model.weights.h5'
    model, tokenizer = load_model(checkpoint_path)
    while True:
        input_text = input("Enter input: ")
        response = inference(model, tokenizer, input_text)
        print(response)
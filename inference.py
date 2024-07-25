#This file loads the saved model and runs infrence on it
import tensorflow as tf
from model import TransformerModel as Transformer
import dataLoader


def load_weights(weights_file):
    model = Transformer(num_layers=4, d_model=256, num_heads=4, dff=1024, input_vocab_size=50257, target_vocab_size=50257, pe_input=1000, pe_target=1000, rate=0.1)
    model.load_weights(weights_file)
    return model


def infrence(model, input, tokenizer):
    # Tokenize the input
    input = tokenizer(input, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
    # Get the model prediction
    output = model(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
    # Decode the output
    output = tokenizer.decode(output)
    return output


tokenizer = dataLoader.get_tokenizer()
tokenizer.pad_token = tokenizer.eos_token
model = load_weights("transformer_model.weights.h5")

while True:
    input = input("Enter a sentence: ")
    output = infrence(model, input, tokenizer)
    print(output)
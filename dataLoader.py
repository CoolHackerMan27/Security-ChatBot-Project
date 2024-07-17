from transformers import GPT2Tokenizer
import tensorflow as tf
from tesorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


def readFile(fileName):
    with open(fileName, 'r') as file:
        return file.readlines()

#Use tensorflow Dataset API to create a data pipeline, this is wayyyyy better then what I was doing before
def data_generator(input_file, target_file, batch_size, max_length):
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    input_text = readFile(input_file)
    target_text = readFile(target_file)
    dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text))

    def preprocess_input(input_text, target_text):
        combined_text = tf.strings.join(["Instruction: ", input_text, "\nResponse: ", target_text]) 
        return combined_text
        
    def tokenize_map(text):
        encodings = tokenizer(text, max_length=max_length, truncation=True, padding="max_length")
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        lables = tf.pad(input_ids[1:], [[0, 1]])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataset = dataset.map(preprocess_input, num_prallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(tokenize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(1024).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return tokenizer, dataset

    


def get_data_pipeline(input_file, target_file, batch_size, max_length):
    return data_generator(input_file, target_file, batch_size, max_length)
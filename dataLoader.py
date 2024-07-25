from transformers import GPT2Tokenizer
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


def readFile(fileName):
    with open(fileName, 'r') as file:
        return file.readlines()

# Use tensorflow Dataset API to create a data pipeline, this is wayyyyy better then what I was doing before


def data_generator(input_file, target_file, batch_size, max_length):
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    input_text = readFile(input_file)
    target_text = readFile(target_file)
    dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text))

    def preprocess_input(input_text, target_text):
        combined_text = tf.strings.join(
            ["Instruction: ", input_text, "\nResponse: ", target_text])
        return combined_text

    def tokenize_map(text):
        text_str = text.numpy().decode('utf-8')
        encodings = tokenizer(text_str, max_length=max_length,
                              truncation=True, padding="max_length")
        input_ids = tf.constant(encodings["input_ids"], dtype=tf.int32)
        attention_mask = tf.constant(
            encodings["attention_mask"], dtype=tf.int32)
        labels = tf.pad(input_ids[1:], [[0, 1]])
        return input_ids, attention_mask, labels

    dataset = dataset.map(lambda x, y: preprocess_input(x, y))
    dataset = dataset.map(lambda x: tf.py_function(
        func=tokenize_map, inp=[x], Tout=[tf.int32, tf.int32, tf.int32]))

    # dataset = dataset.take(5000).cache().repeat() #This only works if the dataset fits in memory, so small datasets or large memory(like a lot. 128GB+) Using take(K) and repeat() to force the dataset to be fully cached.
    # dataset = dataset.take(5000000)  # Use this for picking smaller subset of the main dataset
    dataset = dataset.shuffle(1024).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return tokenizer, dataset


def get_data_pipeline(input_file, target_file, batch_size, max_length):
    return data_generator(input_file, target_file, batch_size, max_length)

import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.preprocessing.sequence import pad_sequences
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_file, target_file, batch_size, tokenizer):
        self.input_file = input_file
        self.target_file = target_file
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __len__(self):
        return sum(1 for _ in open(self.input_file)) // self.batch_size

    def __getitem__(self, idx):
        with open(self.input_file, 'r') as f_in, open(self.target_file, 'r') as f_tar:
            f_in.seek(idx * self.batch_size)
            f_tar.seek(idx * self.batch_size)
            
            input_batch = [next(f_in).strip() for _ in range(self.batch_size)]
            target_batch = [next(f_tar).strip() for _ in range(self.batch_size)]

        input_encoded = self.tokenizer.tokenize(input_batch).merge_dims(-2, -1).to_tensor()
        target_encoded = self.tokenizer.tokenize(target_batch).merge_dims(-2, -1).to_tensor()

        return input_encoded, target_encoded

def pad_sequences_to_same_length(seq1, seq2, padding='post'):
    max_length = max(tf.shape(seq1)[1], tf.shape(seq2)[1])
    padded_seq1 = tf.pad(seq1, [[0, 0], [0, max_length - tf.shape(seq1)[1]]], constant_values=0)
    padded_seq2 = tf.pad(seq2, [[0, 0], [0, max_length - tf.shape(seq2)[1]]], constant_values=0)
    return padded_seq1, padded_seq2

def tokenize_and_encode(chunk, tokenizer):
    tokenized_text = tokenizer.tokenize(chunk)
    return tokenized_text.merge_dims(-2, -1)

def get_tokenizer(vocab_file='bert_vocab.txt'):
    return text.BertTokenizer(vocab_file, lower_case=True)

def get_data_pipeline(input_file, target_file, batch_size, vocab_file='bert_vocab.txt'):
    tokenizer = get_tokenizer(vocab_file)
    data_generator = DataGenerator(input_file, target_file, batch_size, tokenizer)
    return tokenizer, data_generator
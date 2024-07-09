from transformers import GPT2Tokenizer
import tensorflow as tf

def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_file, target_file, batch_size, tokenizer, max_length=512):
        self.input_file = input_file
        self.target_file = target_file
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return sum(1 for _ in open(self.input_file)) // self.batch_size

    def __getitem__(self, idx):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f_in, open(self.target_file, 'r', encoding='utf-8') as f_tar:
                # Move the file pointer to the correct position
                for _ in range(idx * self.batch_size):
                    next(f_in)
                    next(f_tar)

                input_batch = []
                target_batch = []

                for _ in range(self.batch_size):
                    input_batch.append(next(f_in).strip())
                    target_batch.append(next(f_tar).strip())

                # Combine input and target for instruction-following format
                combined_texts = [f"Instruction: {inp}\nResponse: {tar}" for inp, tar in zip(input_batch, target_batch)]

                # Tokenize
                self.tokenizer.pad_token = self.tokenizer.eos_token
                encoded = self.tokenizer(combined_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="tf")

                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']

                # Create labels (shift input_ids right by 1)
                labels = tf.pad(input_ids[:, 1:], [[0, 0], [0, 1]])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        except (UnicodeDecodeError, StopIteration) as e:
            print(f"Skipping batch {idx} due to an error: {e}")
            return self.__getitem__((idx + 1) % self.__len__())


def get_data_pipeline(input_file, target_file, batch_size, max_length=512):
    tokenizer = get_tokenizer()
    data_generator = DataGenerator(input_file, target_file, batch_size, tokenizer, max_length)
    return tokenizer, data_generator
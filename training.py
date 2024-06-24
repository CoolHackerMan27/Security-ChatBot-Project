import tensorflow as tf
import tensorflow_text as text

# Load your data
with open('train.from', 'r', encoding='utf-8') as f:
    train_inputs = f.readlines()
with open('train.to', 'r', encoding='utf-8') as f:
    train_targets = f.readlines()

# Create a tokenizer
tokenizer = text.BertTokenizer('bert_vocab.txt', lower_case=True)

#Transformer model class
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs, targets, training, enc_padding_mask, 
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inputs, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            targets, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

# Tokenize and encode the data
def tokenize_and_encode(texts):
    return tokenizer.tokenize(texts).to_tensor()
# Custom learning rate scheduler
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp, True, None, None, None)
        loss = loss_function(tar_real, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_inputs_encoded = tokenize_and_encode(train_inputs)
train_targets_encoded = tokenize_and_encode(train_targets)

BUFFER_SIZE = 20000
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((train_inputs_encoded, train_targets_encoded))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Initialize the model
model = TransformerModel(
    num_layers=6, 
    d_model=512, 
    num_heads=8, 
    dff=2048, 
    input_vocab_size=tokenizer.vocab_size(), 
    target_vocab_size=tokenizer.vocab_size(), 
    pe_input=1000, 
    pe_target=1000
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (inp, tar)) in enumerate(dataset):
        batch_loss = train_step(inp, tar)
        total_loss += batch_loss
    
    print(f'Epoch {epoch + 1}, Loss: {total_loss/len(dataset)}')
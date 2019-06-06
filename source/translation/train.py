import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join as path_join
import time
import matplotlib.pyplot as plt

import data_preparation
import net_params_saving

# do the tensor ops instantly
tf.enable_eager_execution()

# --paths--
DATASET_PATH = "..\\..\\translation_dataset\\dataset.txt"

# --dataset--
NUM_EXAMPLES = None  # how many examples to use for training. None = all data
VALIDATION_PCT = 0.00

# --hyperparams--
BATCH_SIZE = 32
EMBEDDING_DIM = 64
HIDDEN_UNITS = 256
EPOCHS = 10

# checkpoints
CHECKPOINT_DIR = 'S:\\transl_models'
CHECKPOINT_PREFIX = "checkpoint"

inputs_train, targets_train, input_words, target_words, max_len_input, max_len_target = \
    data_preparation.load_dataset(DATASET_PATH, NUM_EXAMPLES)

# split training and validation sets using an 80-20 split
inputs_train, inputs_val, targets_train, targets_val = \
    train_test_split(inputs_train, targets_train, test_size=VALIDATION_PCT)

# vocabulary lengths
input_vocab_len = len(input_words.word2idx)
target_vocab_len = len(target_words.word2idx)

# train len and number of batches
inputs_train_len = len(inputs_train)
num_batches = inputs_train_len // BATCH_SIZE

print(
    "Data lengths: "
    f"Input train = {inputs_train_len}, Target train = {len(targets_train)}, "
    f"Input val = {len(inputs_val)}, Target val = {len(targets_val)}, "
    f"Input vocab = {input_vocab_len}, Target vocab = {target_vocab_len}, Train examples = {NUM_EXAMPLES}\n"
    "Hyperparams: "
    f"Batch size = {BATCH_SIZE}, Embedding dim = {EMBEDDING_DIM}, Hidden units = {HIDDEN_UNITS}, Epochs = {EPOCHS}, "
    f"Total batches: {num_batches}"
)

# init tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).shuffle(inputs_train_len)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


def gru(units):
    # Gated recurrent network
    return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


# init encoder and decoder
encoder = Encoder(input_vocab_len, EMBEDDING_DIM, HIDDEN_UNITS, BATCH_SIZE)
decoder = Decoder(target_vocab_len, EMBEDDING_DIM, HIDDEN_UNITS, BATCH_SIZE)

# optimizer
optimizer = tf.train.AdamOptimizer()


# loss function
def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


checkpoint_prefix = path_join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

min_total_loss = np.Inf
total_time = time.time()

for epoch in range(EPOCHS):
    start = time.time()

    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inputs, targets)) in enumerate(dataset):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inputs, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([target_words.word2idx['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targets.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targets[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targets[:, t], 1)

        batch_loss = (loss / int(targets.shape[1]))

        total_loss += batch_loss

        variables = encoder.variables + decoder.variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        if batch % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch}, Loss {batch_loss.numpy():.4f}')

    # save the model every time a minimum train loss is found
    if min_total_loss >= total_loss / num_batches:
        min_total_loss = total_loss / num_batches
        print(f"Found new min loss: {min_total_loss}. Saving model...")
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / num_batches))
    print(f'Time taken for this epoch: {time.time() - start:.4f} seconds\n')

print(f'Total train time: {time.time() - total_time:.4f} seconds')


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = data_preparation.preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, HIDDEN_UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp,
                                                max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))
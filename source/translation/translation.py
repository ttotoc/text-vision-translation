import tensorflow as tf
import numpy as np

import params_saveload
import data_preparation
from train import Decoder, Encoder


def evaluate(sentence, encoder, decoder, input_lang_data, target_lang_data, max_len_input, max_len_target):
    attention_plot = np.zeros((max_len_target, max_len_input))

    sentence = data_preparation.preprocess_sentence(sentence)

    inputs = [input_lang_data.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_len_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, encoder.enc_units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang_data.word2idx['<start>']], 0)

    for t in range(max_len_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_lang_data.idx2word[predicted_id] + ' '

        if target_lang_data.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def translate(sentence, encoder, decoder, input_lang_data, target_lang_data, max_len_input, max_len_target):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, input_lang_data, target_lang_data,
                                                max_len_input, max_len_target)

    return result


def perform(sentence):
    from arguments import ARGS

    params = params_saveload.load(ARGS.translation_model["params"])

    encoder = Encoder(params["input_vocab_len"], params["embedding_dim"], params["hidden_units"], params["batch_size"])
    decoder = Decoder(params["target_vocab_len"], params["embedding_dim"], params["hidden_units"], params["batch_size"])
    optimizer = tf.train.AdamOptimizer()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    checkpoint.restore(ARGS.translation_model["model"])

    return translate(
        sentence,
        encoder, decoder,
        params["input_lang_data"], params["target_lang_data"],
        params["max_len_input"], params["max_len_target"]
    )

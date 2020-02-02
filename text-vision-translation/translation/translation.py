import os

import numpy as np
import tensorflow as tf

from translation.train import Decoder, Encoder
from . import data_preparation, params_saveload
from helpers.consts import PATH_MODELS
from configuration.config import get_setting_value
from configuration.settings import TRANSLATION

CURRENT_MODEL, PARAMS, ENCODER, DECODER, OPTIMIZER = None, None, None, None, None


def evaluate(sentence, encoder, decoder, input_lang_data, target_lang_data, max_len_input, max_len_target):
    attention_plot = np.zeros((max_len_target, max_len_input))

    sentence = data_preparation.preprocess_sentence(sentence)

    try:
        inputs = [input_lang_data.word2idx[word] for word in sentence.split(' ')]
    except KeyError as ke:
        print(f"Word not found in model dictionary. Aborting translation...")
        return
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


def perform(sentences):
    config_model = get_setting_value(TRANSLATION)
    model_path = os.path.join(PATH_MODELS, config_model)

    global CURRENT_MODEL, PARAMS, ENCODER, DECODER, OPTIMIZER
    if model_path != CURRENT_MODEL:
        print("[INFO] loading translation model language...")
        PARAMS = params_saveload.load(model_path + ".pickle")

        ENCODER = Encoder(PARAMS["input_vocab_len"], PARAMS["embedding_dim"], PARAMS["hidden_units"],
                          PARAMS["batch_size"])
        DECODER = Decoder(PARAMS["target_vocab_len"], PARAMS["embedding_dim"], PARAMS["hidden_units"],
                          PARAMS["batch_size"])
        OPTIMIZER = tf.train.AdamOptimizer()

        checkpoint = tf.train.Checkpoint(optimizer=OPTIMIZER,
                                         encoder=ENCODER,
                                         decoder=DECODER)

        print("[INFO] loading translation model variables...")
        checkpoint.restore(model_path)

        CURRENT_MODEL = model_path

    print("Translations: ")
    for i, sentence in enumerate(sentences):
        translated = translate(
            sentence,
            ENCODER, DECODER,
            PARAMS["input_lang_data"], PARAMS["target_lang_data"],
            PARAMS["max_len_input"], PARAMS["max_len_target"]
        )

        print(
            f"Input: {sentence}\n"
            f"Output: {translated}\n"
        )

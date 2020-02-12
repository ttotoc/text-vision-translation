import os

import numpy as np
import tensorflow as tf

from configuration.config import get_setting_value
from configuration.settings import TRANSLATION
from helpers.consts import PATH_MODEL_TRANSLATION
from translation.train import Decoder, Encoder
from translation import data_preparation, params_saveload

CURRENT_MODEL, PARAMS, ENCODER, DECODER, OPTIMIZER = None, None, None, None, None


def translate(sentence, encoder, decoder, input_lang_data, target_lang_data, max_len_input, max_len_target):
    sentence = data_preparation.preprocess_sentence(sentence)

    try:
        inputs = [input_lang_data.word2idx[word] for word in sentence.split()]
    except KeyError as ke:
        print(f"Word not found in the model's dictionary. Translation aborted.")
        return

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_len_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, encoder.enc_units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang_data.word2idx['<start>']], 0)

    for t in range(max_len_target):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_lang_data.idx2word[predicted_id] + ' '

        if target_lang_data.idx2word[predicted_id] == '<end>':
            return result

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def perform(sentences):
    model_path = os.path.join(PATH_MODEL_TRANSLATION, get_setting_value(TRANSLATION))

    global CURRENT_MODEL, PARAMS, ENCODER, DECODER, OPTIMIZER
    if model_path != CURRENT_MODEL:
        print("[INFO] loading translation model language...")
        PARAMS = params_saveload.load(model_path + ".pickle")

        ENCODER = Encoder(PARAMS["input_vocab_len"], PARAMS["embedding_dim"], PARAMS["hidden_units"],
                          PARAMS["batch_size"])
        DECODER = Decoder(PARAMS["target_vocab_len"], PARAMS["embedding_dim"], PARAMS["hidden_units"],
                          PARAMS["batch_size"])
        OPTIMIZER = tf.compat.v1.train.AdamOptimizer()

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

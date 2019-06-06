import tensorflow as tf

import unicodedata
import re
from sortedcontainers import SortedSet
from numpy import Inf


def unicode_to_ascii(text):
    # remove nonspacing marks
    return ''.join(
        ch for ch in unicodedata.normalize('NFD', text)
        if unicodedata.category(ch) != 'Mn'
    )


def preprocess_sentence(text):
    # convert to ascii, lower and strip
    text = unicode_to_ascii(text).lower().strip()

    # add a space between a word and the punctuation following it
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)

    # replace everything with space except letters, ".", "?", "!" and ","
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    # remove any leading and trailing whitespace
    text = text.strip()

    # add a start and an end token to the sentence so that the model know when to start and stop predicting
    text = '<start> ' + text + ' <end>'

    return text


def build_dataset(path, num_examples=None):
    # returns a list of 'num_examples' [english, romanian] sentence list pairs

    if num_examples:
        lines = open(path, encoding='UTF-8').readlines()[:num_examples]
    else:
        lines = open(path, encoding='UTF-8').readlines()

    sentence_pairs = [
        [preprocess_sentence(w) for w in line.split('\t')]
        for line in lines
    ]

    return sentence_pairs


# class that stores data about the words of a language
# word2idx(ex: "dog" -> 1) and idx2word(ex: 1 -> "dog") dictionaries
# and a set with all the words
class LanguageData:

    def __init__(self, sentences):

        # self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}

        vocabulary = SortedSet()
        for phrase in sentences:
            vocabulary.update(phrase.split(' '))

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(vocabulary):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def convert_sentences(sentences, lang_data):
    max_len = -Inf
    converted_sentences = []
    for sentence in sentences:
        conv_sentence = [lang_data.word2idx[word] for word in sentence.split()]
        if max_len < len(conv_sentence):
            max_len = len(conv_sentence)
        converted_sentences.append(conv_sentence)

    return converted_sentences, max_len


def load_dataset(path, num_examples=None):
    sentence_pairs = build_dataset(path, num_examples)

    # get the sentence pairs
    input_sentences = [en for en, _ in sentence_pairs]
    target_sentences = [ro for _, ro in sentence_pairs]

    input_lang_data = LanguageData(input_sentences)
    target_lang_data = LanguageData(target_sentences)

    # store the sentences as a 2d array of word indices and calculate max sentence length along the way
    inputs, max_len_input = convert_sentences(input_sentences, input_lang_data)
    targets, max_len_target = convert_sentences(target_sentences, target_lang_data)

    # add padding to every sentence until length reaches max_len_#
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        inputs,
        maxlen=max_len_input,
        padding='post'
    )

    targets = tf.keras.preprocessing.sequence.pad_sequences(
        targets,
        maxlen=max_len_target,
        padding='post'
    )

    return inputs, targets, input_lang_data, target_lang_data, max_len_input, max_len_target

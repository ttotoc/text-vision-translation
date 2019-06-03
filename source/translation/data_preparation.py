import tensorflow as tf

import unicodedata
import re


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

    return text


def build_dataset(path, num_examples=None):
    # returns a tuple of 'num_examples' (english, romanian) sentence tuples

    if num_examples:
        lines = open(path, encoding='UTF-8').readlines()[:num_examples]
    else:
        lines = open(path, encoding='UTF-8').readlines()

    sentence_pairs = tuple(tuple(preprocess_sentence(w) for w in line.split('\t')) for line in lines)

    return sentence_pairs


# class that stores data about the words of a language
# word2idx(ex: "dog" -> 1) and idx2word(ex: 1 -> "dog") dictionaries
# and a set with all the words
class WordData:

    def __init__(self, sentences):

        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        for phrase in sentences:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_row_len(tensor):
    # given a list of lists, return the max length of a sublist
    return max(len(lst) for lst in tensor)


def load_dataset(path, num_examples=None):
    sentence_pairs = build_dataset(path, num_examples)

    # get the sentence pairs
    en_sentences = tuple(en for en, _ in sentence_pairs)
    ro_sentences = tuple(ro for _, ro in sentence_pairs)

    input_words = WordData(en_sentences)
    target_words = WordData(ro_sentences)

    # store the sentences as a 2d array of word indices
    # input(en sentences)
    input_tensor = tuple(
        tuple(input_words.word2idx[word] for word in sentence.split()) for sentence in en_sentences
    )
    # target(ro sentences)
    target_tensor = tuple(
        tuple(target_words.word2idx[word] for word in sentence.split()) for sentence in ro_sentences
    )

    max_len_input = max_row_len(input_tensor)
    max_len_target = max_row_len(target_tensor)

    # add padding to every sentence until length reaches max_len_#
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        input_tensor,
        maxlen=max_len_input,
        padding='post'
    )

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        target_tensor,
        maxlen=max_len_target,
        padding='post'
    )

    return input_tensor, target_tensor, input_words, target_words, max_len_input, max_len_target

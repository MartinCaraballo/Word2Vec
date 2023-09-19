import tqdm
import tensorflow as tf
from keras import layers
import re
import string
import io
import numpy as np


# BASADO EN EL EJEMPLO DE TENSORFLOW: https://www.tensorflow.org/text/tutorials/word2vec#embedding_lookup_and_analysis

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns + 1)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    """
    BASADO EN EL EJEMPLO DE TENSORFLOW: https://www.tensorflow.org/text/tutorials/word2vec#embedding_lookup_and_analysis
    Generates skip-gram pairs with negative sampling for a list of sequences (int-encoded sentences) based on
    window size, number of negative samples and vocabulary size.
    """
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def vectorize_sentences(training_file, vocab_size, sequence_length, autotune):
    """
    :param training_file: file with the data to train the model
    :param vocab_size: vocabulary size
    :param sequence_length: number of words in a sequence
    :param autotune: tf data autotune
    :return: vectorize_layer, inverse_vocab, sequences
    """
    # Use the non-empty lines to construct a tf.data.TextLineDataset object for the next steps:
    text_ds = tf.data.TextLineDataset(training_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    # Now, create a custom standardization function to lowercase the text and
    # remove punctuation.
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase,
                                        '[%s]' % re.escape(string.punctuation), '')

    # Use the `TextVectorization` layer to normalize, split, and map strings to
    # integers. Set the `output_sequence_length` length to pad all samples to the
    # same length.
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Call TextVectorization.adapt on the text dataset to create vocabulary.
    vectorize_layer.adapt(text_ds.batch(1024))

    # Save the created vocabulary for reference. Once the state of the layer has been adapted to represent the text
    # corpus, the vocabulary can be accessed with TextVectorization.get_vocabulary. This function returns a list of
    # all vocabulary tokens sorted (descending) by their frequency.
    inverse_vocab = vectorize_layer.get_vocabulary()

    # Vectorize the data in text_ds.
    text_vector_ds = text_ds.batch(1024).prefetch(autotune).map(vectorize_layer).unbatch()

    # To prepare the dataset for training a word2vec model, flatten the dataset into a list of sentence vector
    # sequences. This step is required as you would iterate over each sentence in the dataset to produce positive and
    # negative examples.
    sequences = list(text_vector_ds.as_numpy_iterator())
    for seq in sequences[:5]:
        print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

    return vectorize_layer, inverse_vocab, sequences


def performance_config(autotune, batch_size, buffer_size, targets, contexts, labels):
    """
    Configure the dataset for performance To perform efficient batching for the potentially large number of
    training examples, use the tf.data.Dataset API. After this step, you would have a tf.data.Dataset object of (
    target_word, context_word), (label) elements to train your word2vec model!
    """
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    print(dataset)

    dataset = dataset.cache().prefetch(buffer_size=autotune)
    print(dataset)

    return dataset


def word2id_dicc(vocab):
    id2word = {}
    for index, word in enumerate(vocab):
        id2word[word] = index

    return id2word


def get_similar_words(word, word2id_vocab, id2word_vocab, weights, number_of_words=10):
    word_id = word2id_vocab[word]
    if word_id is None:
        return f'{word} not found.'

    word_vec = weights[word_id]
    similarities = []

    for other_word in id2word_vocab:
        if other_word is not word:
            other_word_vec = weights[word2id_vocab[other_word]]
            sim_val = np.dot(word_vec, other_word_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(other_word_vec))
            if sim_val > 0.2: similarities.append((other_word, sim_val))

    return sorted(similarities, key=lambda x: x[1], reverse=True)[0:number_of_words]


def export_to_tsv(vectors_file_path, metadata_file_path, vocab, weights):
    """
    Exports the model to tsv files, that can be open at 'Embedding Projector':
    https://projector.tensorflow.org/?_gl=1*17jl0e0*_ga*MTY4ODgzNTc0My4xNjk1MDg0ODA4*_ga_W0YLR4190T*MTY5NTA4NDgwOC4xLjEuMTY5NTA4Nzc3MS4wLjAuMA..
    :param vectors_file_path:
    :param metadata_file_path:
    :param vocab: model vocabulary dictionary
    :param weights: words vectors dictionary
    :return:
    """
    out_v = io.open(vectors_file_path, 'w', encoding='utf-8')
    out_m = io.open(metadata_file_path, 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()

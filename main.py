import tensorflow as tf
from word2vec import *
import numpy as np

if __name__ == '__main__':
    AUTOTUNE = tf.data.AUTOTUNE
    SEED = 42

    training_file = tf.keras.utils.get_file('shakespeare.txt',
                                   'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
                                   )

    # Define the vocabulary size and the number of words in a sequence.
    vocab_size = 4096
    sequence_length = 10

    vectorize_layer, inverse_vocab, sequences = vectorize_sentences(training_file, vocab_size, sequence_length, AUTOTUNE)

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2,
        num_ns=4,
        vocab_size=vocab_size,
        seed=SEED
    )

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000

    dataset = performance_config(AUTOTUNE, BATCH_SIZE, BUFFER_SIZE, targets, contexts, labels)

    # Building model
    embedding_dim = 128
    word2vec = Word2Vec(vocab_size, embedding_dim, 4)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    # Define a callback to log training statistics for TensorBoard:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    # Train the model on the dataset for some number of epochs:
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    word2id_vocab = word2id_dicc(vocab)

    word = 'king'
    print(f'Similar to {word}: {get_similar_words(word, word2id_vocab, vocab, weights, 20)}')

    # export_to_tsv('vectors.tsv', 'metadata.tsv', vocab, weights)



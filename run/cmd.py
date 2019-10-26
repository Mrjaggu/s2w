"""MLP - machine-learning-production

Usage:
    mlp-cli train <dataset-dir> <model-file> [--vocab-size=<vocab-size>]
    mlp-cli ask <model-file> <question>
    mlp-cli (-h | --help)

Arguments:
    <dataset-dir>  Directory with dataset.
    <model-file>   Serialized model file.
    <question>     Text to be classified.

Options:
    --vocab-size=<vocab-size>  Vocabulary size. [default: 10000]
    -h --help                  Show this screen.

"""
import os

from docopt import docopt
from sklearn.metrics import classification_report

from mlp import DumbModel, Dataset

def train_model(dataset_dir, model_file, vocab_size):
    print(f'Training model from directory {dataset_dir}')
    print(f'Vocabulary size: {vocab_size}')

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    dset = Dataset(train_dir, test_dir)
    X, y = dset.get_train_set()

    model = DumbModel(vocab_size=vocab_size)
    model.train(X, y)

    print(f'Storing model to {model_file}')
    model.serialize(model_file)

    X_test, y_test = dset.get_test_set()
    y_pred = model.predict(X_test)

def ask_model(model_file, question):
    print(f'Asking model {model_file} about "{question}"')

    model = DumbModel.deserialize(model_file)

    y_pred = model.predict([question])
    print(y_pred[0])


def main():
    arguments = docopt(__doc__)

    if arguments['train']:
        train_model(arguments['<dataset-dir>'],
                    arguments['<model-file>'],
                    int(arguments['--vocab-size'])
        )
    elif arguments['ask']:
        ask_model(arguments['<model-file>'],
                  arguments['<question>'])

if __name__ == '__main__':
    main()

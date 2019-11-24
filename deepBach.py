"""
@author: Gaetan Hadjeres
"""

import click
import csv

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DeepBach.model_manager import DeepBach
from DeepBach.helpers import *
from grader.grader import score_chorale
from tqdm import tqdm
from grader.grader import plot_distributions
from itertools import islice


@click.command()
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=16,
              help='number of training epochs')
@click.option('--train', is_flag=True,
              help='train the specified model for num_epochs')
@click.option('--num_iterations', default=500,
              help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')
@click.option('--model_id', default=0,
              help='ID of the model to train and generate from')
def main(note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         linear_hidden_size,
         batch_size,
         num_epochs,
         train,
         num_iterations,
         sequence_length_ticks,
         model_id,
         ):
    print(f'Model ID: {model_id}')

    dataset_manager = DatasetManager()

    print('step 1/5: initialize empty metadata')
    metadatas = [
       FermataMetadata(),
       TickMetadata(subdivision=4),
       KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4
    }

    print('step 2/5: load pre-existing dataset or generate new dataset')
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(name='bach_chorales', **chorale_dataset_kwargs)

    dataset = bach_chorales_dataset

    print('step 3/5: create model architecture')
    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size,
        model_id=model_id,
    )

    if train:
        print('step 4/5: train model')
        deepbach.train(batch_size=batch_size,
                       num_epochs=num_epochs)
    else:
        print('step 4/5: load model')
        deepbach.load()
        deepbach.cuda()

    # generate chorales
    print('step 5/5: score chorales')
    dataset.calculate_histograms()
    chorale_scores = {}
    generation_scores = {}

    print('Scoring real chorales')
    count = 5
    iterator = islice(dataset.iterator_gen(), count)
    for chorale_id, chorale in tqdm(enumerate(iterator)):
        note_score, rhythm_score = score_chorale(chorale, dataset)
        chorale_scores[chorale_id] = (note_score, rhythm_score, note_score + rhythm_score)

    print('Generating and scoring generated chorales')
    ensure_dir(f'generations/{model_id}')
    for i in tqdm(range(count)):
        chorale, tensor_chorale, tensor_metadata = deepbach.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
        )
        chorale.write('midi',  f'generations/{model_id}/c{i}.mid')
        note_score, rhythm_score = score_chorale(chorale, dataset)
        generation_scores[i] = (note_score, rhythm_score, note_score + rhythm_score)

    with open('data/chorale_scores.csv', 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        for key, value in chorale_scores.items():
            reader.writerow([key, *value])

    with open('data/generation_scores.csv', 'w') as generation_file:
        reader = csv.writer(generation_file)
        for key, value in generation_scores.items():
            reader.writerow([key, *value])

    plot_distributions('data/chorale_scores.csv', 'data/generation_scores.csv')


if __name__ == '__main__':
    main()

"""
@author: Gaetan Hadjeres
"""

import click
import csv

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager, all_datasets
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from DatasetManager.helpers import GeneratedChoraleIteratorGen

from DeepBach.model_manager import DeepBach
from DeepBach.helpers import *
from grader.grader import score_chorale
from tqdm import tqdm
from grader.grader import plot_distributions
from itertools import chain, islice
import music21
import numpy as np
import pickle


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
@click.option('--update', is_flag=True,
              help='update the specified model for update_iterations')
@click.option('--num_iterations', default=500,
              help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')
@click.option('--model_id', default=0,
              help='ID of the model to train and generate from')
@click.option('--update_iterations', default=2,
              help='number of iterations of generating chorales, scoring, and updating trained model')
@click.option('--num_generations', default=2,
              help='number of chorales to generate at each iteration')
@click.option('--include_transpositions', is_flag=True,
              help='whether to include transpositions (for dataset creation, or for pointing to the right folder at generation time)')
def main(note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         linear_hidden_size,
         batch_size,
         num_epochs,
         train,
         update,
         num_iterations,
         sequence_length_ticks,
         model_id,
         update_iterations,
         num_generations,
         include_transpositions,
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
        'subdivision':    4,
        'include_transpositions': include_transpositions,
    }

    print('step 2/5: load pre-existing dataset or generate new dataset')
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(name='bach_chorales',
                                                                        **chorale_dataset_kwargs)
    dataset = bach_chorales_dataset
    histograms_file = 'grader/bach_histograms.txt'
    if os.path.exists(histograms_file):
        print('Loading Bach chorale histograms')
        with open(histograms_file, 'rb') as fin:
            dataset.histograms = pickle.load(fin)
    else:
        dataset.calculate_histograms()
        with open(histograms_file, 'wb') as fo:
            pickle.dump(dataset.histograms, fo)

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
        print('step 4/5: train base model')
        deepbach.train(batch_size=batch_size,
                       num_epochs=num_epochs)
    else:
        print('step 4/5: load model')
        deepbach.load()
        deepbach.cuda()

    if update:
        print(f'step 4/5: update base model over {update_iterations} iterations')
        for i in range(update_iterations):
            print(f'Iteration {i}')
            picked_chorales = []
            num_picked_chorales = 0
            for j in tqdm(range(num_generations)):
                chorale, tensor_chorale, tensor_metadata = deepbach.generation(
                    num_iterations=num_iterations,
                    sequence_length_ticks=sequence_length_ticks,
                )

                score, scores = score_chorale(chorale, dataset)
                if score < 0.5:
                    picked_chorales.append(chorale)
                    num_picked_chorales += 1

            print(f'Number of picked chorales: {num_picked_chorales}')
            all_datasets.update({f'generated_chorales_{i}': {'dataset_class_name': ChoraleDataset,
                                                             'corpus_it_gen': GeneratedChoraleIteratorGen(
                                                                 picked_chorales)}})
            generated_dataset: ChoraleDataset = dataset_manager.get_dataset(name=f'generated_chorales_{i}',
                                                                            index2note_dicts=dataset.index2note_dicts,
                                                                            note2index_dicts=dataset.note2index_dicts,
                                                                            voice_ranges=dataset.voice_ranges,
                                                                            **chorale_dataset_kwargs)

            deepbach.dataset = generated_dataset
            deepbach.train(batch_size=batch_size,
                           num_epochs=2,
                           split=[1, 0],                # use all selected chorales for training
                           early_stopping=False)

    # generate chorales
    print('step 5/5: score chorales')
    # chorale_scores = {}
    generation_scores = {}
    gen_count = 20

    # print('Scoring real chorales')
    # smaller_iterator = islice(dataset.iterator_gen(), gen_count)
    # for chorale_id, chorale in tqdm(enumerate(smaller_iterator)):
    #     score, scores = score_chorale(chorale, dataset)
    #     chorale_scores[chorale_id] = (*scores, score)

    print('Generating and scoring generated chorales')
    ensure_dir(f'generations/{model_id}')
    for i in tqdm(range(gen_count)):
        chorale, tensor_chorale, tensor_metadata = deepbach.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=sequence_length_ticks,
        )
        chorale.write('midi',  f'generations/{model_id}/c{i}.mid')
        score, scores = score_chorale(chorale, dataset)
        generation_scores[i] = (*scores, score)

    # threshold = np.mean([value[-1] for value in generation_scores])

    # with open('data/chorale_scores.csv', 'w') as chorale_file:
    #     reader = csv.writer(chorale_file)
    #     for key, value in chorale_scores.items():
    #         reader.writerow([key, *value])

    with open('data/generation_scores.csv', 'w') as generation_file:
        reader = csv.writer(generation_file)
        for key, value in generation_scores.items():
            reader.writerow([key, *value])

    plot_distributions('data/chorale_scores.csv', 'data/generation_scores.csv')


if __name__ == '__main__':
    main()

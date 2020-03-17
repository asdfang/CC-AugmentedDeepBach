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
from experiments.visualize_score_dist import plot_distributions, plot_boxplot_per_iteration
from itertools import islice


weights = {'error': .5,
           'parallel_error': .15,
           'note': 5,
           'rhythm': 1,
           'directed_interval': 20}


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
@click.option('--include_transpositions', is_flag=True,
              help='whether to include transpositions (for dataset creation, or for pointing to the right folder at generation time)')
@click.option('--update_iterations', default=100,
              help='number of generation-update iterations')
@click.option('--generations_per_iteration', default=50,
              help='number of chorales to generate at each iteration')
@click.option('--num_generations', default=351,
              help='number of generations, scoring, write to CSV file')
@click.option('--score_chorales', is_flag=True,
              help='score real Bach chorales')
@click.option('--write_scores', is_flag=True,
              help='whether to record scores in csv file')
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
         include_transpositions,
         update_iterations,
         generations_per_iteration,
         num_generations,
         score_chorales,
         write_scores
         ):

    print('step 1/3: prepare dataset')
    dataset_manager = DatasetManager()
    metadatas = [
        FermataMetadata(),
        TickMetadata(subdivision=4),
        KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids': [0, 1, 2, 3],
        'metadatas': metadatas,
        'sequences_size': 8,
        'subdivision': 4,
        'include_transpositions': include_transpositions,
    }

    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(name='bach_chorales',
                                                                        **chorale_dataset_kwargs)
    dataset = bach_chorales_dataset
    load_or_pickle_distributions(dataset)

    print('step 2/3: prepare model')
    print(f'Model ID: {model_id}')
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
        print('step 2a/3: train base model')
        deepbach.train(batch_size=batch_size,
                       num_epochs=num_epochs,
                       split=[0.85, 0.15])
    else:
        print('step 2a/3: load model')
        deepbach.load()
        deepbach.cuda()

    if update:
        print(f'step 2b/3: update base model over {update_iterations} iterations')
        thres = get_threshold('data/chorale_scores.csv', col=-1)
        print(f'Threshold for selection: {thres}')
        update_file = open('data/update_scores.csv', 'w')
        reader = csv.writer(update_file)
        reader.writerow(['iteration', 'chorale ID', 'score'])
        for i in range(update_iterations):
            print(f'----------- Iteration {i} -----------')
            picked_chorales = []
            num_picked_chorales = 0
            ensure_dir(f'generations/{model_id}/{i}')
            for j in tqdm(range(generations_per_iteration)):
                chorale, tensor_chorale, tensor_metadata = deepbach.generation(
                    num_iterations=num_iterations,
                    sequence_length_ticks=sequence_length_ticks,
                )

                score = score_chorale(chorale, dataset)
                
                # write data to csv file
                reader.writerow([i, j, score])      # iteration, generation #, score

                # worst Bach chorale score rounded up to nearest .01
                if score > thres:
                    print(f'Picked chorale {j} with score {score}')
                    picked_chorales.append(chorale)
                    num_picked_chorales += 1

                chorale.write('midi', f'generations/{model_id}/{i}/c{j}.mid')

            print(f'Number of picked chorales for iteration {i}: {num_picked_chorales}')

            if num_picked_chorales == 0:
                continue

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
                           split=[1, 0],  # use all selected chorales for training
                           early_stopping=False)

    # generate chorales    
    if score_chorales:
        chorale_scores = {}
        print('Scoring real chorales')
        for chorale_id, chorale in tqdm(enumerate(dataset.iterator_gen()), total=num_generations):
            score = score_chorale(chorale, dataset)
            chorale_scores[chorale_id] = score
            if chorale_id == num_generations:
                break

        # write scores to file
        if write_scores:
            with open('data/chorale_scores.csv', 'w') as chorale_file:
                reader = csv.writer(chorale_file)
                reader.writerow(['', 'score'] + list(weights.keys()))
                for id, value in chorale_scores.items():
                    reader.writerow([id, value])

    if num_generations != 0:
        generation_scores = {}
        print('Generating and scoring generated chorales')
        ensure_dir(f'generations/{model_id}')
        for i in range(num_generations):
            chorale, tensor_chorale, tensor_metadata = deepbach.generation(
                num_iterations=num_iterations,
                sequence_length_ticks=sequence_length_ticks,
            )
            chorale.write('midi', f'generations/{model_id}/c{i}.mid')
            score = score_chorale(chorale, dataset)
            generation_scores[i] = score

        # write scores to file
        if write_scores:
            with open(f'data/model{model_id}_scores.csv', 'w') as generation_file:
                reader = csv.writer(generation_file)
                reader.writerow(['', 'score'] + list(weights.keys()))
                for id, value in generation_scores.items():
                    reader.writerow([id, value])


if __name__ == '__main__':
    main()

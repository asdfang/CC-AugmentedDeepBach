import sys
sys.path.insert(0, '../')

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

import click
import random
from DeepBach.helpers import ensure_dir
import os, shutil
from tqdm import tqdm


@click.command()
@click.option('--model_ids', nargs=3,
              help='ID of the models to compare')
@click.option('--num_comparisons_per_model', default=10,
              help='number of generations per model')
def main(model_ids, num_comparisons_per_model):
    print(f'Model IDs: {model_ids}')

    dataset_manager = DatasetManager()

    print('step 1/3: prepare dataset of real Bach chorales')
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
        'include_transpositions': False,        # we only want real Bach chorales
    }

    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(name='bach_chorales',
                                                                        **chorale_dataset_kwargs)
    dataset = bach_chorales_dataset
    get_pairs(dataset, model_ids=model_ids, num_comparisons_per_model=num_comparisons_per_model)


def get_pairs(dataset, model_ids=None,
              eval_dir='../generations/paired_evaluation',
              num_comparisons_per_model=None):
    """
    Arguments:
        dataset: dataset of real Bach chorales
        model_ids: list of model IDs we are comparing
        eval_dir: folder for evaluation
        num_comparisons_per_model: number of generations for each model
    """
    real_chorales = random.sample(list(dataset.iterator_gen()), num_comparisons_per_model*len(model_ids))
    generation_files = []
    for model_id in model_ids:
        for i in range(num_comparisons_per_model):
            generation_files.append(('../generations', str(model_id), f'c{i}.mid'))

    random.shuffle(generation_files)

    for i, chorale in tqdm(enumerate(real_chorales), total=num_comparisons_per_model):
        pair_dir = os.path.join(eval_dir, f'{i}')
        ensure_dir(pair_dir)
        chorale.write('midi', f'{pair_dir}/chorale_{i}.mid')
        model_id = generation_files[i][1]
        shutil.copy(os.path.join(*generation_files[i]), f'{pair_dir}/gen_{model_id}_{i}.mid')


if __name__ == '__main__':
    main()

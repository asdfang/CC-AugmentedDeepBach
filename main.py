from grader.grader import score_chorale
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager, all_datasets
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from DatasetManager.helpers import GeneratedChoraleIteratorGen

from DeepBach.model_manager import DeepBach
from DeepBach.helpers import *

print('step 1/3: prepare dataset')
dataset_manager = DatasetManager()
metadatas = [
    FermataMetadata(),
    TickMetadata(subdivision=4),
    KeyMetadata()
]
chorale_dataset_kwargs = {
    'voice_ids': [1, 1, 2, 3],
    'metadatas': metadatas,
    'sequences_size': 8,
    'subdivision': 4,
    'include_transpositions': False,
}

bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(name='bach_chorales',
                                                                    **chorale_dataset_kwargs)
dataset = bach_chorales_dataset
load_or_pickle_distributions(dataset)

print(dataset.gaussian.covariances_)

# chorale = converter.parse('generations/6/c187.mid')
# score = score_chorale(chorale, dataset)
# print(score)

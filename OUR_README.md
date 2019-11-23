```
dataset_manager = DatasetManager()

# initialize empty metadata
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
# load pre-existing dataset
bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
    name='bach_chorales',
    **chorale_dataset_kwargs
    )
dataset = bach_chorales_dataset

train_dataloader, val_dataloader, test_dataloader = bach_chorales_dataset.data_loaders(batch_size=128,
                                                                                       split=(0.85, 0.10))
print('Num Train Batches: ', len(train_dataloader))
print('Num Valid Batches: ', len(val_dataloader))
print('Num Test Batches: ', len(test_dataloader))

# create model architecture
deepbach = DeepBach(
    dataset=dataset,
    note_embedding_dim=note_embedding_dim,
    meta_embedding_dim=meta_embedding_dim,
    num_layers=num_layers,
    lstm_hidden_size=lstm_hidden_size,
    dropout_lstm=dropout_lstm,
    linear_hidden_size=linear_hidden_size
)

# train or load model
if train:
    deepbach.train(batch_size=batch_size,
                   num_epochs=num_epochs)
else:
    deepbach.load()
    deepbach.cuda()

# generate chorales
print('Generation')
score, tensor_chorale, tensor_metadata = deepbach.generation(
    num_iterations=num_iterations,
    sequence_length_ticks=sequence_length_ticks,
)
score.show('txt')

# musescore representation
score.show()
```
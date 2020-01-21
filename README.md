# DeepBach
The code uses python 3.6 together with [PyTorch v1.0](https://pytorch.org/) and
 [music21](http://web.mit.edu/music21/) libraries.

## Usage
Every time we run `deepBach.py`, there are three things we can do: (1) load a model (no `--train` or `--update` flag), (2) train a model (include `--train` flag), (3) update a trained model (include `--update` flag). A model can be trained and updated in one run by including both corresponding flags.

```
Usage: deepBach.py [OPTIONS]

Options:
[general]
  --train                           train or retrain the specified model
  --update                          update the specified model
  --model_id INTEGER                ID of the model to load, train, and/or update
  --include_transpositions          whether to include transpositions when training a model/
                                    whether the trained model includes transpositions when updating or generating
  --help                            Show this message and exit.
[model hyperparameters]
  --note_embedding_dim INTEGER      size of the note embeddings
  --meta_embedding_dim INTEGER      size of the metadata embeddings
  --num_layers INTEGER              number of layers of the LSTMs
  --lstm_hidden_size INTEGER        hidden size of the LSTMs
  --dropout_lstm FLOAT              amount of dropout between LSTM layers
  --linear_hidden_size INTEGER      hidden size of the Linear layers
  --batch_size INTEGER              training batch size
  --num_epochs INTEGER              number of training epochs
[update parameters]
  --update_iterations               number of generation-update iterations
  --generations_per_iteration       number of chorales to generate at each iteration
[generation parameters]
  --num_generations                 number of generations for inspection
  --num_iterations INTEGER          number of parallel pseudo-Gibbs sampling iterations
  --sequence_length_ticks INTEGER   length of the generated chorale (in ticks)
```

## Experimental pipeline
### Train base models
We train two models: (1) a model trained on the original chorales and their transpositions, which is the setting equivalent to the original DeepBach paper, and (2) a model without transpositions, which serves as our base model. We will train until we reach the lowest validation loss.
- Train base model with transpositions
```
python deepBach.py --include_transpositions --model_id=10
```

- Train base model without transpositions
```
python deepBach.py --train --model_id=11
```

### Update models
- First, copy the base model so that we do not overwrite it. 
```
cp -r 11 12
```
- Update the base model
```
python deepBach.py --update --update_iterations=50 --generations_per_iteration=50
```

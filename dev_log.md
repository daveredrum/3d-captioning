# Development Log

## Aug. 9

__progress:__
- performance of cross-modality retrieval on __ShapeNetCore__:

<table>
  <tr>
    <td rowspan=2 align="center">Method</td>
    <td colspan=3 align="center">Shape-Text</td>
    <td colspan=3 align="center">Text-Shape</td>
  </tr>
  <tr>
    <td align="center">RR@1</td>
    <td align="center">RR@5</td>
    <td align="center">NDCG@5</td>
    <td align="center">RR@1</td>
    <td align="center">RR@5</td>
    <td align="center">NDCG@5</td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td align="center">0.07</td>
    <td align="center">0.34</td>
    <td align="center">0.06</td>
    <td align="center">0.11</td>
    <td align="center">0.35</td>
    <td align="center">0.23</td>
  </tr>
  <tr>
    <td>Text2Shape</td>
    <td align="center">0.83</td>
    <td align="center">3.37</td>
    <td align="center">0.73</td>
    <td align="center">0.40</td>
    <td align="center">2.37</td>
    <td align="center">1.35</td>
  </tr>
  <tr>
    <td><b>ours (32)</b></td>
    <td align="center"><b>1.68</b></td>
    <td align="center"><b>6.57</b></td>
    <td align="center"><b>1.53</b></td>
    <td align="center"><b>0.95</b></td>
    <td align="center"><b>4.52</b></td>
    <td align="center"><b>2.71</b></td>
  </tr>
</table>

> __best models:__
>
> |resolution|train_size|test_size|learning_rate|weight_decay|batch_size|random|
> |---|---|---|---|---|---|---|
> |32|-1|-1|2e-4|5e-4|100|False|

## Jun. 21

__progress:__
- performance of shape captioning on __ShapeNetCore__:

|Model|BLEU-1|BLEU-2|BLEU-3|BLEU-4|CIDEr|
|---|---|---|---|---|---|
|Baseline|0.573|0.314|0.151|0.075|0.117|
|__FC__|0.731|0.502|0.306|0.180|0.290|
|__att2in__|0.771|0.548|0.341|0.202|0.329|
|__att2all__|0.783|0.563|0.355|0.214|0.339|
|__spatial__|0.775|0.551|0.344|0.206|0.333|
|__adaptive__|0.781|0.561|0.354|0.214|0.344|

> __best models:__
>
> |Model|train_size|test_size|learning_rate|weight_decay|batch_size|beam_size|dropout|
> |---|---|---|---|---|---|---|---|
> |__FC__|-1|-1|2e-4|0|100|1|0|
> |__att2in__|-1|-1|2e-4|1e-5|256|1|0|
> |__att2all__|-1|-1|2e-4|1e-5|256|1|0|
> |__spatial__|-1|-1|2e-4|1e-4|256|1|0|
> |__adaptive__|-1|-1|2e-4|1e-4|256|1|0|

- performance of shape captioning on __Primitives__:

|Model|BLEU-1|BLEU-2|BLEU-3|BLEU-4|CIDEr|
|---|---|---|---|---|---|
|Baseline|0.867|0.781|0.681|0.569|0.550|
|__Text2Shape<br/>LSTM__|0.889|0.819|0.772|0.727|0.755|

> __best models:__
>
> |Model|train_size|test_size|learning_rate|weight_decay|batch_size|beam_size|dropout|
> |---|---|---|---|---|---|---|---|
> |__Text2Shape<br/>LSTM__|-1|-1|2e-4|0|512|1|0|

## Jun. 19

__progress:__
- migrate image captioning to [image-captioning](https://github.com/daveredrum/image-captioning)
- new data interface for pretrained shape embeddings
- set up baseline for captioning:

|Model|BLEU-1|BLEU-2|BLEU-3|BLEU-4|CIDEr|
|---|---|---|---|---|---|
|Baseline (Nearest neighbor)|0.569|0.314|0.143|0.064|0.096|

__TODOs:__
- finish data interface
- set up captioning baseline with pretrained shape embeddings
- train decoder on pretrained embeddings

## Jun. 12

__progress:__
- performance evaluation:

|Model|BLEU-1|BLEU-2|BLEU-3|BLEU-4|CIDEr|
|---|---|---|---|---|---|
|Baseline (Nearest neighbor)|0.48|0.281|0.166|0.1|0.383|
|__ResNet152 <br/> LSTM__|__0.720__|__0.536__|__0.388__|__0.286__|__0.805__|
|__ResNet152 <br/> Attention <br/> LSTM__|__0.697__|__0.504__|__0.351__|__0.249__|__0.718__|
|NeuralTalk2|0.625|0.45|0.321|0.23|0.66|
|Show and Tell|0.666|0.461|0.329|0.27|-|
|Show, Attend and Tell|0.707|0.492|0.344|0.243|-|
|Adaptive Attention|0.742|0.580|0.439|0.266|1.085|
|Neural Baby Talk|0.755|-|-|0.347|1.072|

> __best models:__
>
> |Model|train_size|test_size|learning_rate|weight_decay|batch_size|beam_size|dropout|
> |---|---|---|---|---|---|---|---|
> |__ResNet152 <br/> LSTM__|-1|-1|2e-4|0|512|7|0|
> |__ResNet152 <br/> Attention <br/> LSTM__|-1|-1|2e-4|1e-5|128|7|0|

## Jun. 8

> __milestone:__ beam search is now available

__progress:__
- offline feature extraction for boosting the training
- constrained the pretrained encoder as vgg16_bn and resnet101
- enabled beam search
- performance evaluation:

|Model|BLEU-1|BLEU-2|BLEU-3|BLEU-4|CIDEr|
|---|---|---|---|---|---|
|Baseline (Nearest neighbor)|0.48|0.281|0.166|0.1|0.383|
|__ResNet101 + LSTM__|__0.567__|__0.356__|__0.207__|__0.133__|__0.241__|
|__ResNet101 + Attention + LSTM__|__0.57__|__0.363__|__0.212__|__0.134__|__0.259__|
|NeuralTalk2|0.625|0.45|0.321|0.23|0.66|
|Show and Tell|0.666|0.461|0.329|0.27|-|
|Show, Attend and Tell|0.707|0.492|0.344|0.243|-|

__TODOs:__
- search for the best hyperparameters
- evaluation with different beam search sizes

## May.25

__progress:__
- fixed attention, but the attention weights don't change much on small dataset
- applied new architecture
- omitted validation loss, refer to NLP metrics in validation phase
- generating of text in validation phase won't stop until __<END>__
- added a script to visualize the attention weights, upsampling is applied

__TODOs:__
- investigate why the attention weights don't change much
- implement beam decoder

## May.16

__progress:__
- added encoder with attention, more work needed to be done

__TODOs:__
- modularize the attention mechanism
- implement a new LSTM module to make it compatiable with attended visual context
- refine the encoder to make it compatiable with attention

## May.15

__progress:__
- fixed bug with cross entropy loss by specifying ignore_index=0 as 0 is only used for padding the sequences
- refined the status report
- constrainted the dictionary size, currently using 5000 as maximum

__TODOs:__
- add encoder with attention

## May.14

__progress:__
- added VGG16 with batch normalization as encoder
- modified the last few layers of the encoders
- refined the script for testing
- added tensorboard support

__TODOs:__
- train models on full training set
- try more powerful encoders, e.g. ResNet50
- try to use text2shape pretrained embeddings

## May.10

> __milestone:__ pretrained model for image captioning are now available

__progress:__
- enabled implicit switch for teacher forcing in solver
- added interface for specify pretrained model
- refined the preprocessing scripts to resize the images to the desired sizes, both for ShapeNetCore and for MSCOCO
- refined the title for saved images, all necessary parameters included

__TODOs:__
- figure out how to document experiments results and automatic synchronization
- refine the testing script for captioning MSCOCO 

## May.8

__progress:__
- fixed the bug with `solver.py`, now a new epoch in front of all epochs records losses before training, which also works as the sanity check for the training and validation losses
- added data interface for MSCOCO dataset
- added the preprocessing script for MSCOCO dataset
- refine the proprocessing script for ShapeNetCore dataset
- excluded METEOR for now

__TODOs:__
- generalize the model
- train the model on MSCOCO dataset

## May.7

__progress:__
- added more metrics: BLEU-n, CIDEr, ROUGE_L
- enabled plotting the metrics above

__TODOs:__
- generalize the model
- find a better implementation of METEOR

## May.3

__progress:__
- fixed bug with the `train.py`
- visualized the results in jupyter notebook, see more details in google docs

__TODOs:__
- generalize the model
- optimize the data loading step

## May.2

__progress:__
- rewrote the data interface
- fixed bug with the `sample()` method of decoder
- refined the calculation of BLEU score
- refined the training report

__TODOs:__
- fine-tune the model
- fix the bug with `train.py`

## Apr.30

__progress:__
- added BLEU score as one of the evaluation metrics
- added the time tracking for forward and backward pass
- refined the epoch report

__TODOs:__
- fix the bug with BLEU score regarding different split of datasets. (calculations with train/valid/test should be performed on respective corpus instead of on the corpus of the whole dataset)
- fix the bug with text generator to support multi-batches

## Apr.27

> __milestone:__ 3D captioning for shapes are now possible

__progress:__
- successfully deployed the SSTK on the server and ran text2shape project on it
- built 3D encoder and overfit the 3D encoder-decoder
- enabled specifying model types in training script

__TODOs:__
- fix the bug with text generator to support multi-batches


## Apr.23
__progress:__
- overfit the model
- fixed bugs with transformed csv data. Now all captions in the csv file are sorted by the Caption class
- refined the training script

__TODOs:__
- fine-tune encoder-decoder with more data
- try 3d shapes rendering in text2shape project

## Apr.22
__progress:__
- altered the encoder
- trained the pipeline end-to-end
- fixed bugs of the output layer of decoder and the loss function

__TODOs:__
- fine-tune encoder-decoder with more data
- try out more powerful encoder

## Apr.20

> __milestone:__ built encoder-decoder pipeline, online training and fine-tuning is now possible

__progress:__
- built encoder-decoder pipeline
- built solver for encoder-decoder
- modified the dataset for a better training performance
- visualized the training results in `data/`
- the model can now generate captions for images

__TODOs:__
- fine-tune encoder-decoder
- plan for the next stage

## Apr.18

> still on offline stage, i.e. the decoder takes saved visual context vectors as visual inputs

__progress__:
- finished solver for decoder, of which the performance still need improving
- rewrote dataset for captions
- refined the decoder model using `pack_padded_sequence()`

__TODOs__:
- fine-tune decoder
- build text generator
- build captioning pipeline


## Apr.17
extracted visual context vectors and started to build decoder

__TODOs__:
- refine decoder
- complete solver for decoder

## Apr.16
created interface for the csv file which allows to:
- preprocess the descriptions in csv
- initialize dictionaries of descriptions, both from words to indices and from indices to words
- transform descriptions from sentences to lists of indices
- check the sanity of the transformation, i.e. to check if the transformation is reversable

__TODOs__: 
- refine the data type of transformed descriptions, temporarily considering numpy array
- build text generator

## Apr.15
added preprocessing to:
- convert all letters to lowercase
- pad before all punctuations
- add end symbol __<END>__ to the end of each captions (start symbol is unnecessary)

__TODOs__: 
- build dictionary for captions
- build text generator

## Apr.10
built a simple conv net to classify chairs and tables

__TODOs__: 
- add preprocessing
- build dictionary for captions
- build text generator

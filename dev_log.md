# Development Log

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

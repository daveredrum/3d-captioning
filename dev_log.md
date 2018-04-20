# Development Log

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

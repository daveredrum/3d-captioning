# 3d-captioning
## dev_log

### Apr.16
created interface for the csv file which allows to:
- preprocess the descriptions in csv
- initialize dictionaries of descriptions, both from words to indices and from indices to words
- transform descriptions from sentences to lists of indices
- check the sanity of the transformation, i.e. to check if the transformation is reversable

__TODOs__: 
- refine the data type of transformed descriptions, temporarily considering numpy array
- build text generator

### Apr.15
added preprocessing to:
- convert all letters to lowercase
- pad before all punctuations
- add end symbol __<END>__ to the end of each captions (start symbol is unnecessary)

__TODOs__: 
- build dictionary for captions
- build text generator

### Apr.10
built a simple conv net to classify chairs and tables

__TODOs__: 
- add preprocessing
- build dictionary for captions
- build text generator

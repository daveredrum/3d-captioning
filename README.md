# 3d-captioning
## dev_log

### Api.15
added preprocessing to:
- convert all letters to lowercase
- pad before all punctuations
- add end symbol __<END>__ to the end of each captions (start symbol is unnecessary)

__TODO__: 
- build dictionary for captions
- build text generator

### Apr.10
built a simple conv net to classify chairs and tables

__TODO__: 
- add preprocessing
- build dictionary for captions
- build text generator

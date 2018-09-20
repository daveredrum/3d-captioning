# Development Log

## Sep. 7

__progress:__
- performance of cross-modality retrieval on __ShapeNetCore__:

> comparison of models: <br/>
>  <br/>
> __Text2Shape__: original joint-embedding method without attention <br/>
> __Text2Shape-32__: replicated text2shape without attention on 32 <br/>
> __Text2Shape-64__: replicated text2shape without attention on 64 <br/>
> __noattention-64__: new model without attention on 64 <br/>
> __self_nosep-64__: original self-attention module, spatial and channel attentions are not seperated <br/>
> __self_sep-64__: original self-attention module, spatial and channel attentions are seperated <br/>
> __selfnew_nosep-64__: similarity-based self-attention module, spatial and channel attentions are not seperated <br/>
> __selfnew_sep_cf-64__: similarity-based self-attention module, spatial and channel attentions are seperated and stacked. (channel first) <br/>
> __selfnew_sep_p-64__: similarity-based self-attention module, spatial and channel attentions are seperated and parallel <br/>

<table>
  <tr>
    <td rowspan=2 align="center">Arch</td>
    <td colspan=3 align="center">Shape-Text</td>
    <td colspan=3 align="center">Text-Shape</td>
    <td rowspan=2 align="center">Total</td>
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
    <td align="center">1.16</td>
  </tr>
  <tr>
    <td>Text2Shape</td>
    <td align="center">0.83</td>
    <td align="center">3.37</td>
    <td align="center">0.73</td>
    <td align="center">0.40</td>
    <td align="center">2.37</td>
    <td align="center">1.35</td>
    <td align="center">9.05</td>
  </tr>
  <tr>
    <td>Text2Shape-32</td>
    <td align="center">0.27</td>
    <td align="center">2.96</td>
    <td align="center">0.54</td>
    <td align="center">0.55</td>
    <td align="center">3.13</td>
    <td align="center">1.85</td>
    <td align="center">9.31</td>
  </tr>
  <tr>
    <td>Text2Shape-64</td>
    <td align="center">0.61</td>
    <td align="center">3.43</td>
    <td align="center">0.69</td>
    <td align="center">0.70</td>
    <td align="center">2.77</td>
    <td align="center">1.69</td>
    <td align="center">9.88</td>
  </tr>
  <tr>
    <td>noattention-64</td>
    <td align="center">0.47</td>
    <td align="center">3.50</td>
    <td align="center">0.65</td>
    <td align="center">0.77</td>
    <td align="center">3.24</td>
    <td align="center">1.97</td>
    <td align="center">10.59</td>
  </tr>
  <tr>
    <td>self_nosep-64</td>
    <td align="center">0.67</td>
    <td align="center">1.95</td>
    <td align="center">0.45</td>
    <td align="center">0.52</td>
    <td align="center">2.34</td>
    <td align="center">1.41</td>
    <td align="center">7.34</td>
  </tr>
  <tr>
    <td>self_sep-64</td>
    <td align="center">0.74</td>
    <td align="center">3.30</td>
    <td align="center">0.74</td>
    <td align="center">0.52</td>
    <td align="center">2.80</td>
    <td align="center">1.64</td>
    <td align="center">9.74</td>
  </tr>
  <tr>
    <td>selfnew_nosep-64</td>
    <td align="center">0.40</td>
    <td align="center">2.89</td>
    <td align="center">0.55</td>
    <td align="center">0.42</td>
    <td align="center">2.31</td>
    <td align="center">1.38</td>
    <td align="center">7.96</td>
  </tr>
  <tr>
    <td>selfnew_sep_p-64</td>
    <td align="center">1.10</td>
    <td align="center">4.64</td>
    <td align="center">1.02</td>
    <td align="center">0.73</td>
    <td align="center">3.75</td>
    <td align="center">2.21</td>
    <td align="center">13.42</td>
  </tr>
  <tr>
    <td>selfnew_sep_sf-64</td>
    <td align="center">1.14</td>
    <td align="center">5.05</td>
    <td align="center">1.10</td>
    <td align="center">0.87</td>
    <td align="center">4.53</td>
    <td align="center">2.68</td>
    <td align="center">15.37</td>
  </tr>
  <tr>
    <td>selfnew_sep_cf-64</td>
    <td align="center"><b>1.62</b></td>
    <td align="center"><b>6.53</b></td>
    <td align="center"><b>1.49</b></td>
    <td align="center"><b>1.09</b></td>
    <td align="center"><b>5.08</b></td>
    <td align="center"><b>3.03</b></td>
    <td align="center"><b>18.84</b></td>
  </tr>
</table>

> __best models:__
>
> |arch|train_size|val_size|learning_rate|weight_decay|batch_size|epoch|random|length|
> |---|---|---|---|---|---|---|---|---|
> |noattn|-1|-1|2e-4|5e-4|100|20|False|18|
> |attn|-1|-1|2e-4|5e-4|100|20|False|96|

- performance of shape captioning on __ShapeNetCore__:

> comparison of models: __Text2Shape/SSAM__<br/>
>  <br/>
> __Text2Shape__: using Text2Shape embeddings for training <br/>
> __SSAM__: using selfnew_sep_cf embeddings for training <br/>

|Model|BLEU-1|BLEU-2|BLEU-3|BLEU-4|CIDEr|
|---|---|---|---|---|---|
|Baseline|0.573|0.314|0.151|0.075|0.117|
|__FC__|0.748/0.754|0.522/0.528|0.347/0.354|0.235/0.239|0.394/0.402|
|__att2in__|0.742/0.758|0.520/0.536|0.350/0.363|0.240/0.253|0.414/0.434|
|__att2all__|0.742/-|0.521/-|0.354/-|0.247/-|0.432/-|
|__adaptive__|0.731/0.746(?)|0.506/0.525(?)|0.344/0.354(?)|0.237/0.245(?)|0.429/0.429(?)|

> __best models:__
>
> |Model|train_size|test_size|learning_rate|weight_decay|batch_size|beam_size|dropout|
> |---|---|---|---|---|---|---|---|
> |__FC__|-1|-1|1e-4|1e-5|256|1|0|
> |__att2in__|-1|-1|1e-4|1e-5|256|1|0|
> |__att2all__|-1|-1|1e-4|1e-5|256|1|0|
> |__adaptive__|-1|-1|1e-4|1e-5|256|1|0|

<s>
## Aug. 23

__progress:__
- performance of cross-modality retrieval on __ShapeNetCore__:

<table>
  <tr>
    <td rowspan=2 align="center">Method</td>
    <td colspan=3 align="center">Shape-Text<br/>(Text2Shape-64/ours-64)</td>
    <td colspan=3 align="center">Text-Shape<br/>(Text2Shape-64/ours-64)</td>
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
    <td>Full-MM</td>
    <td align="center">1.68/<b>77.48</b></td>
    <td align="center">6.57/<b>88.34</b></td>
    <td align="center">1.53/<b>79.41</b></td>
    <td align="center">0.95/<b>51.45</b></td>
    <td align="center">4.52/<b>91.29</b></td>
    <td align="center">2.71/<b>73.32</b></td>
  </tr>
</table>

> __best models:__
>
> |train_size|test_size|learning_rate|weight_decay|batch_size|epoch|random|
> |---|---|---|---|---|---|---|
> |-1|-1|2e-4|5e-4|32|10|False|
</s>
  
## Aug. 9

__progress:__
- performance of cross-modality retrieval on __ShapeNetCore__:

<table>
  <tr>
    <td rowspan=2 align="center">Method</td>
    <td colspan=3 align="center">Shape-Text<br/>(Text2Shape/ours-32/ours-64)</td>
    <td colspan=3 align="center">Text-Shape<br/>(Text2Shape/ours-32/ours-64)</td>
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
    <td>LBA-MM</td>
    <td align="center">0.07/0.07/<b>0.13</b></td>
    <td align="center">0.37/0.20/<b>0.67</b></td>
    <td align="center">0.07/0.04/<b>0.14</b></td>
    <td align="center">0.08/0.08/<b>0.11</b></td>
    <td align="center">0.34/0.40/<b>0.40</b></td>
    <td align="center">0.21/0.24/<b>0.27</b></td>
  </tr>
  <tr>
    <td>ML</td>
    <td align="center">0.13/<b>0.13</b>/0.06</td>
    <td align="center">0.47/<b>0.67</b>/0.40</td>
    <td align="center">0.11/<b>0.13</b>/0.07</td>
    <td align="center">0.13/<b>0.13</b>/0.11</td>
    <td align="center">0.61/0.60/<b>0.64</b></td>
    <td align="center">0.36/0.36/<b>0.38</b></td>
  </tr>
  <tr>
    <td>Full-TST</td>
    <td align="center"><b>0.94</b>/0.20/0.54</td>
    <td align="center"><b>3.69</b>/1.40/3.22</td>
    <td align="center"><b>0.85</b>/0.22/0.66</td>
    <td align="center">0.22/0.32/<b>0.70</b></td>
    <td align="center">1.63/1.26/<b>3.10</b></td>
    <td align="center">0.87/0.79/<b>1.86</b></td>
  </tr>
  <tr>
    <td>Full-MM</td>
    <td align="center">0.83/<b>1.68</b>/1.48</td>
    <td align="center">3.37/<b>6.57</b>/3.95</td>
    <td align="center">0.73/<b>1.53</b>/0.98</td>
    <td align="center">0.40/<b>0.95</b>/0.79</td>
    <td align="center">2.37/<b>4.52</b>/3.46</td>
    <td align="center">1.35/<b>2.71</b>/2.07</td>
  </tr>
</table>

> __best models:__
>
> |resolution|train_size|test_size|learning_rate|weight_decay|batch_size|epoch|random|
> |---|---|---|---|---|---|---|---|
> |32|-1|-1|2e-4|5e-4|100|20|False|

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

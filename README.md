# Me trying to improve context2vec

**context2vec: Learning Generic Context Embedding with Bidirectional LSTM**  
Oren Melamud, Jacob Goldberger, Ido Dagan. CoNLL, 2016 [[pdf]](http://u.cs.biu.ac.il/~melamuo/publications/context2vec_conll16.pdf).

## Requirements

* Python 3.6
* Chainer 4.2 ([chainer](http://chainer.org/))
* NLTK 3.0 ([NLTK](http://www.nltk.org/))  - optional (only required for the AWE baseline and MSCC evaluation)

For training you will need the stanford POS tagger https://nlp.stanford.edu/software/tagger.shtml

Run the code from the main directory

## Quick-start
For trained model please contact me

```
python context2vec/eval/explore_context2vec.py MODEL_NAME.params
>> this is a [] book
```
* This will embed the entire sentential context 'this is a \_\_ book' and will output the top-10 target words whose embeddings are closest to that of the context.

## Training a new context2vec model

Rmove all _ signs from your corpus file

Run Stanford POS tagger on the corpus file

* CORPUS_FILE needs to contain your learning corpus with one sentence per line and tokens separated by spaces.
* Run:
```
python context2vec/train/corpus_by_sent_length.py CORPUS_FILE [max-sentence-length]
```
* This will create a directory CORPUS_FILE.DIR that will contain your preprocessed learning corpus
* Run:
```
python context2vec//train/train_context2vec.py -i CORPUS_FILE.DIR  -w  WORD_EMBEDDINGS -m MODEL  -c lstm --deep yes -t 3 --dropout 0.0 -u 300 -e 10 -p 0.75 -b 100 -g 0
```
* This will create WORD_EMBEDDINGS.targets file with your target word embeddings, a MODEL file, and a MODEL.params file. Put all of these in the same directory MODEL_DIR and you're done.

  
NOTE:   
* The current code lowercases all corpus words
* Use of a gpu and mini-batching is highly recommended to achieve good training speeds


## Evaluation

### Microsoft Sentence Completion Challenge (MSCC)

* Download the train and test datasets from [[here]](https://www.microsoft.com/en-us/research/project/msr-sentence-completion-challenge/).

* Run:  
```
python context2vec/eval/sentence_completion.py Holmes.machine_format.questions.txt Holmes.machine_format.answers.txt RESULTS_FILE MODEL_NAME.params
```



## License

Apache 2.0







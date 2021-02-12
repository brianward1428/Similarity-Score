# Processome Node Similarity

## Introduction

This tool is designed for the Center for Complex Network Research (CCNR) at Northeastern University by Brian Ward, under Babak Ravandi. This tool was designed with the ultimate goal to find similar nodes (or text-based labels of nodes) of the FoodOn database, which holds many (~20k) nodes representing food items and food processes.

###  Method of Comparison
This tool uses a [Gensim Word2Vec model](https://radimrehurek.com/gensim/models/word2vec.html) which calculates cosine similarity of vectorized words.  Users may use the default model (trained on the food domain) or train a model of their own to target a different domain.  

### What Type of data can be compared?
This tool is designed to compare **short structured text**. For example find the similarity between *'macintosh apple (baked)'* vs.  *'red delicious apple (pureed)'*. This tool is also designed to treat parenthesized text separately as it likely holds a different type of information (e.g. state of food item).  

### What's Included?

**Pre-Processing**
This tool includes a text-preprocessing method, which includes:
* Basic string cleaning
* Removal of a custom set of words
* Removal of stop words (defaults to [nltk's english stop words](https://gist.github.com/sebleier/554280)  )
* Locating of common bigrams
* Separation of parenthetical text

**Model Training**
This tool comes with a default (food domain trained) model, with the option of training a new model to target a different domain. This method includes :
* Option to start with pre-trained base, which is trained on a wikipedia dump ([ text8 data](http://mattmahoney.net/dc/textdata))
* Full customization of Gensim Word2Vec model.
* Ability to train and build vocabulary on multiple data sources.

**Label Comparison**
There are two methods for label comparison:
1. **sim(...)**, will compare two labels and return the similarity value.
2. **simAll(...)**, will allow for an input of many labels and will compare each pair of labels.
	* includes an optional threshold, to only record pairs with a simVal greater (or equal) to the given threshold.
	* results will be returned in a pandas dataFrame, with the option to include original labels.
* Both have the option to compare parenthesized text separately, as well as a weight to apply the parenthesized text simVal (more below).
* Both have options to customize a weight to the effect of unused words (more below).

## Tool Methods

1. **ProcessStrings(...)** : preprocessing tool for labels.
2.  **trainModel(...)**: creates a Gensim Word2Vec Model  for label comparison.
3. **Sim(...)**: calculate similarity value between two labels
4. **SimAll(...)** : calculate similarity value between each label given.



# processStrings(...)
**FUNCTION :** processStrings(...)  

**INPUT:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **labels:** (list), list of labels or strings to be processed for comparisons. [ Required Param ]   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - If element of list is not of type = string it will be ignored.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **removeParenthesis:** (boolean), if True parenthesis will be removed and processed seperately. [default = False]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **termsToBeRemoved:** (list), list of predetermined words to be removed. [default = [] ]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **removeStopWords:** (boolean), if True given set of stopwords will be removed. [ default = True ]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **stopWords:** (set), set of stopwords to be removed.[default = [nltk's english stop words](https://gist.github.com/sebleier/554280)  )  ]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **locateBigrams:** (boolean), if True will locate and concat  bigrams with given minimum count. [ default = False ]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **bigramMinCount:** (int), minimum count for a bigram to appear to be processed as a bigram (combined) [default=5]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Ex:** if adjacent terms ['artificial', 'sweetener'] are found > 5 times, all instances will be concatenated   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to ['artificial sweetener']  

**OUTPUT:** pandas dataframe containing raw input labels, cleaned labels, and parenthesis labels (if removeParenthesis = True)  

**DESCRIPTION:** this is a pre-processing step to get data ready for the model training and comparisons.  

### Example Usage
    NS = NodeSim()

    # hand-picked terms to be ignored..
    termsToBeRemoved = ["(efsa foodex2)","(efsa foodex)","and similar","probably", "other","food","(us cfr)","(gs gpc)","products", "product","obsolete"]

    labels = NS.processStrings(labels = slim, removeParentheses = False,
                               termsToBeRemoved = termsToBeRemoved, removeStopWords = True,
                               locateBigrams = True, bigramMinCount = 5)

The resulting pandas DF will look like:
![removeParentheses = False](https://i.imgur.com/PQRQY4q.png)

If we were to choose to remove parenthesis (i.e removeParentheses = True) The output would also contain a column for the text within the parentheses:
![removeParentheses = False](https://i.imgur.com/TdUrOhO.png)

### More on Parentheses
This tool is designed to work on finding similarities between structured text labels. The removal of parenthesized text is done under the assumption that information within the parentheses hold a different type of information. By separating the two types of information, we can use our Word2Vec model to calculate a similarity value for the information within the parentheses separately, this value can then be applied to the similarity value with a user-given weight (more on this in next sections).  
#### When should parenthesized text be removed?
When processing data for model training it is recommended to keep label data intact (i..e not removing parenthesized text). Due to the nature of word embedding, where word vectors are determined by looking at surrounding words, trying to train the model on the list of parenthesized data might not add too much value (as many are single or few words). On the other hand there might be clear relationships between text inside/outside parenthesis. In this example we can see that ['guava', 'dried'] gives some context to guava and to dried, and might give us a better similarity to ['apple', 'dried'] if that label were to also exist.

After the model has been created, it would then be the users choice to calculate the two types of information separately or not ( more in next sections) . This would be step where separating the parenthesized text would be necessary.


---
# trainModel(...)


**FUNCTION :** trainModel(...)

**INPUT:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **includeText8:** (boolean), if True, text8 corpus (wikipedia word dump) will be used in model training. [default = True]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - it is recommended to use the text 8 corpus unless you have a large enough corpus to &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   successfully train the model.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -  [More on text8 data](http://mattmahoney.net/dc/textdata)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **corpora:** (list) list of documents of which the model will be trained.     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Corpora should be a list of documents, which should be a list of sentences, which  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   should be a list of words.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - An example input of a single document might look like : [[['the', 'cat', 'in', 'the', &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   'hat'],['the', 'grinch']]]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - These documents should be pre-processed.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **epochsForTraining:** (list), (int), number of epochs desired for model training   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **word2VecArgs:** (dictionary), dictionary of Word2Vec model arguments please see [Gensim documentation](https://radimrehurek.com/gensim/models/word2vec.html)  for argument details.

**OUTPUT:** Gensim Word2Vec Model   

**DESCRIPTION:** this function will allow for basic training of a Gensim Word2Vec model. please see [Gensim documentation](https://radimrehurek.com/gensim/models/word2vec.html) for more details on model training.

### Example Usage

    NS = NodeSim()

    word2VecArgs = {
    'min_count': 2,
    'size' : 100,
    'workers': 3,
    'window' : 5,
    'sg' : 1
    }

    model = NS.trainModel(includeText8 = True,
						  corpora = [labels, recipes, grocery],
					      epochsForTraining = 10,
					      word2VecArgs = word2VecArgs)




# sim(...)
**FUNCTION :** sim (...)

**INPUT:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **model:** Gensim Word2Vec Model to be used for comparison      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **considerParenthesis:** (boolean) if True we will consider terms within parenthesis separately. [default= False]     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **termsA:** (list), list of strings for label_A, if considering parenthesis, do not include words within parenthesis.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **termsB:** (list), list of strings for label_B, if considering parenthesis, do not include words within parenthesis.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **termsA_insideP:**  (list), list of strings inside parenthesis from label A. [default = [  ]]     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **termsB_insideP:** (list), list of strings inside parenthesis from label B. [default = [  ]]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **pWeight:** (float)  0.0-1.0  weight of which to apply to information within the parenthesis. [default=0.1]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **considerRemoved:** (boolean), if True, the ratio of removed terms will be considered. [Default = True]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **removedWeight:** (float) 0.0-1.0 weight for how the removed terms should affect the sim value. [default=0.1]  

**OUTPUT:** (float) 0.0 - 1.0 : similarity value

**DESCRIPTION:** this function will take the given model and use it to compare the two labels.

### Example Usage

    NS = NodeSim()

	simVal2 = NS.sim(model = model,
				termsA = labels['clean'][13],
				termsB = labels['clean'][22],
				termsA_insideP = labels['parentheses'][13],
                termsB_insideP = labels['parentheses'][22],
                pWeight = 0.1,
                considerParentheses = True,
                considerRemoved = True,
                removedWeight = 0.1)

    # now we can take a look at the results:
    print("sim : '{}' vs '{}' = {}".format(labels['raw'][13], labels['raw'][22], simVal2))
**output :**
sim : 'pudding sugar-free instant' vs 'fruit sherbet (artificially sweetened)' = 0.7262362271



### More on pWeight Parameter
As discussed earlier if we are going to compare the information within the parentheses separately, then we will need to allow for a weight at which to apply the parenthesized similarity Value. The pWeight is the parameter which allows the user to adjust this value to fit specific data or intentions.
Here is an example of how the pWeight is applied:

* label A raw = 'fruit sherbet (artificially sweetened)'
* label B raw = 'fruit sherbet (freeze dried)'


```
simVal = 1.0 = sim(['fruit', 'sherbet'], ['fruit', 'sherbet'])
simValParenthesis = 0.3 = sim(['artificially', 'sweetened'], ['freeze' 'dried'])

weightedVal = (simVal * (1 - pWeight)) + (simValParenthesis * pWeight)
```


if pWeight = 1.0 :
```
weightedVal = 0.3 = (1.0 * (1 - 1.0)) + (0.3 * 1.0)
```
if pWeight = 0.5 :
```
weightedVal = 0.65 = (1.0 * (1 - 0.5)) + (0.3 * 0.5)
```
if pWeight = 0.0 :
```
weightedVal = 1.0 = (1.0 * (1 - 0.0)) + (0.3 * 0.0)
```

### More on removedWeight Parameter
As Word2Vec models have a finite dictionary, and extremely sparse words are often ignored in model training, there will likely be a point during two label comparisons where some of the words from one or both of the labels are not in the model's dictionary and must be ignored. The removedWeight parameter allows for the adjustment of calculated simVal based on the ratio of used words : total words.
Here is an example of how the removedWeight is applied:
* label A raw = 'ziao-z pastry'
* label B raw = 'puff pastry'

Now lets assume that 'ziao-z', and 'puff' are both not in the word2vec dictionary.
* then our sim comparison will only compare 'pastry' x 'pastry'
```
simVal = 1.0 = sim(['pastry'], ['pastry'])

P_removed = 0.5 = 1 - ((len(termsA_used) + len(termsB_used)) / (len(termsA_total) + len(termsB_total)))


weighted = 1.0 - (P_removed * removedWeight)
simVal_Final = simVal * weighted
```
if removedWeight = 1.0 :
```
weighted = 0.5 = 1.0 - (0.5 * 1.0)
simVal_Final 0.5 = 1.0 * 0.5
```
if removedWeight = 0.5 :
```
weighted = 0.75 = 1.0 - (0.5 * 0.5)
simVal_Final 0.75 = 1.0 * 0.75
```
if removedWeight = 0.0 :
```
weighted = 1.0 = 1.0 - (0.5 * 0.0)
simVal_Final 1.0 = 1.0 * 1.0
```

---
# simAll(...)

**FUNCTION :** simAll(...)

**INPUT:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **model:** Gensim Word2Vec Model to be used for comparison      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **labels:** (list), list of labels to be compared.      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - these should be pre-processed, each label should be a list of words.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **parenthesis:** (list), list of parenthetical data from labels to be compared. [default = []]       
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Each i'th element should correspond to the i'th label from labels.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - These should be pre-processed, each element of list should be a list of words.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **considerParenthesis:** (boolean) if True we will consider terms within parenthesis separately. [default= False]     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **pWeight:** (float)  0.0-1.0  weight of which to apply to parenthetical sim values. [default = 0.1]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **considerRemoved:** (boolean), if True, the ratio of removed terms will be considered. [Default = True]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **removedWeight:** (float) 0.0-1.0 weight for how the removed terms should affect the sim value. [default=0.1]   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **includeLabels:** (boolean) If True, resulting pandas dataFrame will include the raw labels. [default= False]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **includeThreshold:** (boolean) If True, only similarities greater or equal to the given threshold will be added to the output.     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **threshold:**  (float)  Only sim values >= given threshold will be added to the output. [default=0.9]    

**OUTPUT:** pandas dataFrame with all the compared values, of given labels.     

**DESCRIPTION:** this function will compare all the given labels and will output a pandas DF with calculated similarity values.  

### Example Usage


```
THRESHOLD = 0.99
results = NS.simAll(model,
					labels['clean'].tolist(),
					labels['parentheses'].tolist(),
					pWeight = 0.1,
					removedWeight = 0.01,
					includeLabels = True,
					includeThreshold = True,
					threshold = THRESHOLD)

```
**Example Output:**
![enter image description here](https://i.imgur.com/hDQgUcr.png)



---

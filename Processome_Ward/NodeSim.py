
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from icecream import ic
from gensim.models import Word2Vec
import gensim.downloader as api



'''
##########################################################################################
                                    Helper Functions
##########################################################################################
'''


'''
Clean String
'''
def cleanString(s : str, termsToBeRemoved : list):

    s = s.lower()

    # remove termsToBeRemoved
    for term in termsToBeRemoved:
        s = s.replace(term.lower(), "")

    # remove non-alphabet characters
    myPunc = '!"#$%&\'()*+-./:;,<=>?@[\\]^_`{|}~'
    s = s.translate(s.maketrans(myPunc, ' '*len(myPunc)))
    s = s.translate(s.maketrans(string.digits, ' '*len(string.digits)))

    # clean up extra white-space
    s = re.sub('\s+',' ',s)

    return s.strip()


'''
Remove Stop Words
'''
def filterStopWords(sent : list, stopWords):
    # this could be taking a long time
    # stopWordsEnglish = set(stopwords.words('english'))

    newSent = []

    for word in sent:
        if word not in stopWords:
            newSent.append(word)

    return newSent

'''
Find Bigrams
'''
def findBigrams(corpus : list, min_count : int):

    corpusPhrased = []
    # this actually finds the bigrams
    phrases = Phrases(corpus, min_count = min_count, delimiter=b' ')

    for sent in corpus:
        phrased = phrases[sent]
        corpusPhrased.append(phrased)

    return corpusPhrased


'''
We have a list of terms or strings that we have deemed of no value, so we are going to set a local
variable to represent the strings I found to remove, which are relavent only to the labels.
'''
# DEFAULT_TERMS_TO_REMOVE = ["(efsa foodex2)","(efsa foodex)","and similar","probably","food product","plant","other","food","(us cfr)","(gs gpc)","flavored","product","obsolete"]
STOP_WORDS_ENGLISH = set(stopwords.words('english'))


'''
##########################################################################################
                                    Class Declaration
##########################################################################################
'''

class NodeSim:
    def __init__(self):
        pass

    '''
    FUNCTION : processStrings()
    INPUT:  labels: (list), list of labels or strings to be processed for comparisons. [ Required Param ]
                NOTE: if element of list is not of type = string it will be ignored.
            removeParentheses: (boolean), if True parentheses will be removed and processed seperately. [ default = False ]
            termsToBeRemoved: (list), list of predetermined words to be removed. [default = [] ]
            removeStopWords: (boolean), if True given set of stopwords will be removed. [ default = True ]
            stopWords : (set), set of stopwords to be removed. [default = NLTK english stopwords. link : https://gist.github.com/sebleier/554280]
            locateBigrams : (boolean), if True will locate and concat  bigrams with given minimum count. [ default = False ]
            bigramMinCount : (int), minimum count for a bigram to appear to be processed as a bigram (combined) [default = 5]
                EX: if adjacent terms ['artificial', 'sweetener'] are found > 5 times, all instances will be concatenated to ['artificial sweetener']
    OUTPUT: pandas dataframe containing raw input labels, cleaned labels, (and parentheses labels if removeParentheses = True)
    DESCRIPTION: this will be a string pre-processing step to get data ready for the model comparisons.
    '''
    def processStrings(self, labels, removeParentheses: bool = False, termsToBeRemoved :list = [], removeStopWords: bool = True, stopWords : set = STOP_WORDS_ENGLISH, locateBigrams: bool = False, bigramMinCount: int = 5):

        if removeParentheses:

            clean = []
            parentheses = []

            for term in labels:
                # quick type check:
                if (type(term) == str):
                    # find and seperate info in parentheses
                    par = re.findall('\(.*\)', term, flags=0)
                    # find all will create a list so we just need to turn them into a string
                    pars = " ".join(par)
                    s = re.sub('\(.*\)', '', term)

                    # Clean the labels
                    par = cleanString(pars, termsToBeRemoved)
                    s = cleanString(s, termsToBeRemoved)


                    # Tokenize the labels
                    par = par.split(' ')
                    s = s.split(' ')

                    # Remove StopWords
                    # - going to run this only on the non-empty strings to save time.
                    if par != ['']:
                        if removeStopWords:
                            par = filterStopWords(par, stopWords)
                        parentheses.append(par)
                    else:
                        parentheses.append([])

                    if s != ['']:
                        if removeStopWords:
                            par = filterStopWords(par, stopWords)
                        clean.append(s)
                    else:
                        clean.append([])

                else:
                    clean.append([])
                    parentheses.append([])



            '''
            Now that we have our strings mostly processed we just need to look for bigrams
            '''
            if locateBigrams:
                clean = findBigrams(clean, bigramMinCount)
                parentheses = findBigrams(parentheses, bigramMinCount)

            # now we'll just make a df and return the values
            return pd.DataFrame({'raw': labels, 'clean': clean, 'parentheses': parentheses})

            '''
            We are NOT going to treat the parentheses seperately.
            '''
        else :

            clean = []

            for term in labels:
                # do a quick type check, (th)
                if (type(term) == str):

                    # Clean the labels
                    s = cleanString(term, termsToBeRemoved)

                    # Tokenize the labels
                    s = s.split(' ')

                    if s != ['']:
                        clean.append(s)
                    else:
                        clean.append([])
                else:
                    clean.append([])

            # find bigrams
            if locateBigrams:
                clean = findBigrams(clean, bigramMinCount)

            # now we'll just make a df and return the values
            return pd.DataFrame({'raw': labels, 'clean': clean})


    '''
    FUNCTION : SimHelper
    INPUT:  model: gensim Word2Vec Model to be used for comparison
            termsA: list of strings to be compared
            termsB: list of strings to be compared
            considerRemoved: boolean, if True, the ratio of removed terms will be considered. [Default = True]
            removedWeight: 0.0 - 1.0 weight for how the removed terms should affect the
    OUTPUT: Similarity Value : 0.0 - 1.0
    DESCRIPTION: this function will take the given model and use it to compare the two given lists of terms.
    '''
    def simHelper(self, model : Word2Vec, termsA : list, termsB : list, considerRemoved : bool = False, removedWeight: float = 0.1):

        # check that removedWeight is on the scale of 0.0 - 1.0
        if removedWeight < 0.0 or removedWeight > 1.0:
            raise ValueError("removedWeight parameter must be 0.0 - 1.0")

        if (len(termsA) > 0 and len(termsB) > 0):
            i_terms = []
            # we are only going to consider terms in the models vocabulary.
            for term in termsA:
                if term in model.wv:
                    i_terms.append(term)
            j_terms = []
            # we are only going to consider terms in the models vocabulary.
            for term in termsB:

                if term in model.wv:
                    j_terms.append(term)

            # make sure neither terms are empty
            if (len(i_terms) > 0 and len(j_terms) > 0):

                # maybe this should be scaled down... (Almost Definitely)
                if considerRemoved:

                    '''
                    Now we want to account for words being removed but we need to be careful:
                        - for labels with very few terms, this ratio could get very small.
                        - we will allow user input to weight this ratio, with param: removedWeight
                    '''

                    P_removed = 1 - ((len(i_terms) + len(j_terms)) / (len(termsA) + len(termsB)))
                    P_weighted = 1.0 - (P_removed * removedWeight)

                    return  (model.wv.n_similarity(i_terms, j_terms) * P_weighted)
                else :
                    return model.wv.n_similarity(i_terms, j_terms)
            else :
                return 0.0
        else:
            return 0.0

    '''
    FUNCTION : Sim
    INPUT:  model: gensim Word2Vec Model to be used for comparison
            termsA: (list), list of strings for label_A, if considering parentheses, do not include words within parentheses.
            termsB: (list), list of strings for label_B, if considering parentheses, do not include words within parentheses.
            considerParentheses: (boolean) if True we will consider terms within parentheses seperately. [default= False]
                Note:   - terms within parentheses must be included in related argument.
            termsA_insideP: (list), list of strings inside parentheses from label A. [default = []]
            termsB_insideP: (list), list of strings inside parentheses from label B. [default = []]
            pWeight: (float)  0.0 - 1.0  weight of which to apply to information within the parentheses. [default = 0.1]
            considerRemoved: (boolean), if True, the ratio of removed terms will be considered. [Default = True]
            removedWeight: (float) 0.0 - 1.0 weight for how the removed terms should affect the similarity value. [default = 0.1]
    OUTPUT: (float) 0.0 - 1.0 : similarity value
    DESCRIPTION: this function will take the given model and use it to compare the two labels, treating the information
                inside the parentheses separately and weighted.
    '''
    def sim(self, model : Word2Vec, termsA: list, termsB : list, considerParentheses : bool = False, termsA_insideP : list = [], termsB_insideP : list = [], pWeight : float = 0.1, considerRemoved : bool = False, removedWeight : float = 0.1):

        # check that pWeight is on the scale of 0.0 - 1.0
        if pWeight < 0.0 or pWeight > 1.0:
            raise ValueError("pWeight parameter must be 0.0 - 1.0")

        if removedWeight < 0.0 or removedWeight > 1.0:
            raise ValueError("removedWeight parameter must be 0.0 - 1.0")

        # do we want to consider parentheses separately?
        if considerParentheses:
            simVal = self.simHelper(model, termsA, termsB, considerRemoved, removedWeight)

            # only need to consider this if they both exist.
            if len(termsA_insideP) > 0 and len(termsB_insideP) > 0:
                simVal_P = self.simHelper(model, termsA_insideP, termsB_insideP, considerRemoved, removedWeight)

                weightedVal = (simVal * (1 - pWeight)) + (simVal_P * pWeight)

                return weightedVal

            elif len(termsA_insideP) > 0 or len(termsB_insideP) > 0:
                '''
                So if ony one has anything in the parentheses we should still consider those as removed?
                    - what else could we do?
                    - if we just ignore it than we will end up with a lot of super similar values ( but in our case they should be )
                    - I guess we should think of it the same way as removing words, so we will used the removedWeight.
                '''
                if considerRemoved:
                    '''
                    Now we want to account for words being removed but we need to be careful:
                        - for labels with very few terms, this ratio could get very small.
                        - we will allow user input to weight this ratio, with param: removedWeight
                        - here we will use the total length of words in each label.
                    '''

                    P_removed = 1 - ((len(termsA) + len(termsB)) / (len(termsA) + len(termsA_insideP) + len(termsB) + len(termsB_insideP)))
                    weighted = 1.0 - (P_removed * removedWeight)

                    return  (simVal * weighted)

                else:
                    return simVal


        # we dont care about parentheses.
        return self.simHelper(model, termsA, termsB, considerRemoved, removedWeight)



    '''
    FUNCTION : SimAllNodes
    INPUT:  model: gensim Word2Vec Model to be used for comparison
            labels: (list), list of labels to be compared.
                Note:   - these should be pre-processed, each label should be a list of words.
            parentheses: (list), list of parenthetical data from labels to be compared. [default = []]
                Note:   - each i'th element should correnspond to the i'th label from labels.
                        - These should be pre-processed, each element of list should be a list of words.
            considerParentheses: (boolean) if True we will consider terms within parentheses seperately. [default= False]
                Note:   - terms within parentheses must be included in related argument.
            pWeight: (float)  0.0 - 1.0  weight of which to apply the information within the parentheses. [default = 0.1]
            considerRemoved: (boolean), if True, the ratio of removed terms will be considered. [Default = True]
            removedWeight: (float) 0.0 - 1.0 weight for how the removed terms should affect the similarity value. [default = 0.1]
            includeLabels: (boolean) If True, resulting pandas dataframe will include the raw labels. [default= False]
            includeThreshold: (boolean) If True, only similarities greater or eqaual to the given threshold will be added to the output.
            threshold: (float)  Only similarities greater or equal to the given threshold will be added to the output . [default = 0.9]

    OUTPUT: pandas dataFrame with all the compared values, of given labels.
    DESCRIPTION: this function will compare all the given labels and will output a pandas DF with calculated similarity values.
    '''
    def simAll(self, model, labels : list, parentheses : list = [], considerParentheses : bool = True, pWeight : float = 0.1, considerRemoved : bool = True, removedWeight : float = 0.1, includeLabels : bool = False, includeThreshold : bool = False, threshold : float = 0.9):

        '''
        Argument Checks:
        '''
        if includeThreshold:
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("threshold must be 0.0 - 1.0")

        if pWeight < 0.0 or pWeight > 1.0:
            raise ValueError("pWeight parameter must be 0.0 - 1.0")

        if removedWeight < 0.0 or removedWeight > 1.0:
            raise ValueError("removedWeight parameter must be 0.0 - 1.0")

        if considerParentheses:
            if len(labels) != len(parentheses):
                raise ValueError("labels and parentheses must be the same length")

            Similar = []
            n = len(labels)
            # iterate through each pair (i,j) and calculate their similarity via model.
            for i in range(n):

                i_terms = labels[i]
                i_paren = parentheses[i]

                for j in range(i+1, n):

                    j_terms = labels[j]
                    j_paren = parentheses[j]

                    simVal = self.Sim(model = model, considerParentheses = considerParentheses, termsA = i_terms, termsB = j_terms, termsA_insideP = i_paren, termsB_insideP = j_paren, pWeight = pWeight, considerRemoved = considerRemoved, removedWeight = removedWeight)

                    # check if we care about threshold.
                    if  (not includeThreshold) ^ (simVal >= threshold):
                        # check if we care about labels
                        if includeLabels:
                            Similar.append([" ".join(i_terms) + " (" + " ".join(i_paren) + ")" , i, " ".join(j_terms) + " (" + " ".join(j_paren) + ")", j, simVal ])
                        else:
                            Similar.append([i, j, simVal ])

            if includeLabels:
                return pd.DataFrame(Similar, columns=['label_A','index_A', 'label_B','index_B', 'simVal'])
            else:
                return pd.DataFrame(Similar, columns=['index_A','index_B', 'simVal'])

        else:
            # we dont care about parentheses:
            Similar = []
            n = len(labels)
            # iterate through each pair (i,j) and calculate their similarity via model.
            for i in range(n):
                i_terms = labels[i]

                for j in range(i+1, n):
                    j_terms = labels[j]

                    simVal = self.Sim(model = model, considerParentheses = False, termsA = i_terms, termsB = i_terms, considerRemoved = considerRemoved, removedWeight = removedWeight)

                    # check if we care about threshold.
                    if  (not includeThreshold) ^ (simVal >= threshold):
                        # check if we care about labels
                        if includeLabels:
                            Similar.append([" ".join(i_terms), i, " ".join(j_terms), j, simVal ])
                        else:
                            Similar.append([i, j, simVal ])

            if includeLabels:
                return pd.DataFrame(Similar, columns=['label_A','index_A', 'label_B','index_B', 'simVal'])
            else:
                return pd.DataFrame(Similar, columns=['index_A','index_B', 'simVal'])

    '''
    FUNCTION : trainModelHelper
    INPUT:  model: gensim Word2Vec Model to be used for comparison
            document: list of sentences, sentences should be list of words
            epochs: number of epochs for training.
    OUTPUT: None, model will be trained in place.
    DESCRIPTION: this is a helper function which builds on the vocabulary of a Gensim Word2Vec model.
    '''
    def trainModelHelper(self, model, document, epochs):
        model.build_vocab(document, update=True)
        model.train(document,total_examples=len(document), epochs = epochs)

    '''
    FUNCTION : trainModel
    INPUT:  includeText8: (boolean), if True, text8 corpus (wikipedia word dump) will be used in model training. [default = True]
                Note:   - it is reccomended to use the text 8 corpus unless you have a large enough corpus to successfuly train the model.
                        - More on text8 data : http://mattmahoney.net/dc/textdata
            corpora: (list) list of documents of which the model will be trained.
                Note:   - Corpora should be a list of documents, which should be a list of sentences, which should be a list of words.
                        - An example input of a single document might look like : [[['the', 'cat', 'in', 'the', 'hat'],['the', 'grinch']]]
                        - These documents should be pre-processed.
            epochsForTraining: (int), number of epochs desired for model training
            word2VecArgs: (dictionary), dictionary of Word2Vec model arguments please see Word2Vec for argument details. https://radimrehurek.com/gensim/models/word2vec.html
    OUTPUT: gensim Word2Vec Model
    DESCRIPTION: This function will allow for basic training of a gensim Word2Vec model. please see gensim documentation for more details on
    model training. [https://radimrehurek.com/gensim/models/word2vec.html]
    '''
    def trainModel(self, includeText8: bool = True, corpora = [], epochsForTraining: int = 10, word2VecArgs: dict = {}):
        # some quick checks
        if not includeText8 and len(corpora) < 1:
            raise ValueError("There is no data to train the model, please check your input")

        # should I do a check of the corpora type?
        if len(corpora) > 1:
            if not type(corpora[0]) is list:
                if not type(corpora[0][0]) is list:
                    raise ValueError("Error Invalid Argument : corpora should be a list of documents, documents should be a list of sentences, sentences should be a list of words")
        '''
        Train Model.
        '''
        if includeText8:
            text8 = api.load('text8')

            if len(word2VecArgs) == 0: # if no custom arguments were added.
                model = Word2Vec(text8, min_count=3, size= 100, window =7, sg = 1)
            else:
                model = Word2Vec(text8, **word2VecArgs)
        else:
            # were not going to use text8.
            if len(word2VecArgs) == 0: # if no custom arguments were added.
                model = Word2Vec(corpora[0], min_count=3, size= 100, window =7, sg = 1)
            else:
                model = Word2Vec(corpora[0], **word2VecArgs)

        # now we just need to build and train with included corpera.
        if includeText8:
            for document in corpora:
                self.trainModelHelper(model, document, epochsForTraining)
        else:
            for document in corpora[1:]:
                self.trainModelHelper(model, document, epochsForTraining)


        return model

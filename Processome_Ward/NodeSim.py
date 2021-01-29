
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from icecream import ic
from gensim.models import Word2Vec

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
    INPUT:  labels: list of labels or strings to be processed for model comparisons.
            NOTE: if element of list is not of type string it will be ignored.
            removeParenthesis: default = True. if True parenthesis will be removed and processed seperately.
    OUTPUT: pandas dataframe containing raw input labels, cleaned labels, and parenthesis labels (if removeParenthesis = True)
    DESCRIPTION: this will be a pre-processing step to get data ready for the model comparisons.
    '''
    def processStrings(self, labels, removeParenthesis = True, termsToBeRemoved :list = [], removeStopWords = True, stopWords : set = STOP_WORDS_ENGLISH, locateBigrams = True, bigramMinCount = 5):

        # add our
        # if removeStopWords:
        #     termsToBeRemoved.extend(stopWords)
        # ic(termsToBeRemoved)

        if removeParenthesis:

            clean = []
            parenthesis = []

            for term in labels:
                # do a quick type check, (th)
                if (type(term) == str):
                    # find and seperate info in parenthesis
                    par = re.findall('\(.*\)', term, flags=0)
                    # find all will create a list so we just need to turn them into a string
                    pars = " ".join(par)
                    s = re.sub('\(.*\)', '', term)

                    # print("pars :", pars)

                    # Clean the labels
                    par = cleanString(pars, termsToBeRemoved)
                    s = cleanString(s, termsToBeRemoved)

                    # print("par :", par)
                    # print("type : ", type(par))
                    # Tokenize the labels
                    par = par.split(' ')
                    s = s.split(' ')

                    '''
                    Remove StopWords
                    - going to run this only on the non-empty strings to save time.
                    '''


                    # check that there are real strings
                    if par != ['']:
                        if removeStopWords:
                            par = filterStopWords(par, stopWords)
                        parenthesis.append(par)
                    else:
                        parenthesis.append([])

                    if s != ['']:
                        if removeStopWords:
                            par = filterStopWords(par, stopWords)
                        clean.append(s)
                    else:
                        clean.append([])
                    # ic(clean)
                    # ic(parenthesis)

                else:
                    clean.append([])
                    parenthesis.append([])



            '''
            Now that we have our strings mostly processed we just need to look for bigrams
            '''
            if locateBigrams:
                clean = findBigrams(clean, bigramMinCount)
                parenthesis = findBigrams(parenthesis, bigramMinCount)

                # ic(clean)
                # ic(parenthesis)

            # now we'll just make a df and return the values
            return pd.DataFrame({'raw': labels, 'clean': clean, 'parenthesis': parenthesis})

        # we arent going to treat the parenthesis seperately.
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
    FUNCTION : modelSim
    INPUT:  model: gensim Word2Vec Model to be used for comparison
            termsA: list of strings to be compared
            termsB: list of strings to be compared
            considerRemoved: boolean, if True, the ratio of removed terms will be considered. [Default = True]
            removedWeight: 0.0 - 1.0 weight for how the removed terms should affect the
    OUTPUT: Similarity Value : 0.0 - 1.0
    DESCRIPTION: this function will take the given model and use it to compare the two given lists of terms.
    '''

    def Sim(self, model : Word2Vec, termsA : list, termsB : list, considerRemoved : bool = False, removedWeight: float = 0.5):

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
    FUNCTION : SimWithParenthesis
    INPUT:  model: gensim Word2Vec Model to be used for comparison
            termsA_outsideP: list of strings outside parenthesis from label A
            termsA_insideP: list of strings inside parenthesis from label A
            termsB_outsideP: list of strings outside parenthesis from label B
            termsB_insideP: list of strings inside parenthesis from label B
            pWeight: weight of which to apply to information within the parenthesis
            considerRemoved: boolean, if True, the ratio of removed terms will be considered. [Default = True]
            removedWeight: 0.0 - 1.0 weight for how the removed terms should affect the
    OUTPUT: Similarity Value : 0.0 - 1.0
    DESCRIPTION: this function will take the given model and use it to compare the two labels, treating the information
                inside the parenthesis separately and weighted.
    '''
    def SimWithParenthesis(self, model : Word2Vec, termsA_outsideP : list, termsA_insideP : list, termsB_outsideP : list, termsB_insideP : list, pWeight : float = 0.5, considerRemoved : bool = False, removedWeight : float = 0.5):

        # check that pWeight is on the scale of 0.0 - 1.0
        if pWeight < 0.0 or pWeight > 1.0:
            raise ValueError("pWeight parameter must be 0.0 - 1.0")



        simVal = self.Sim(model, termsA_outsideP, termsB_outsideP, considerRemoved, removedWeight)
        # ic(simVal)
        # only need to consider this if they both exist.
        if len(termsA_insideP) > 0 and len(termsB_insideP) > 0:
            simVal_P = self.Sim(model, termsA_insideP, termsB_insideP, considerRemoved, removedWeight)
            # ic(simVal_P)

            ratioVal = (simVal * (1 - pWeight)) + (simVal_P * pWeight)
            # ic(ratioVal)
            return ratioVal

        else :
            return simVal


    '''
    FUNCTION : SimAllNodes
    ...

    '''
    def SimAllNodes(self, model, labels : list, parenthesis : list, considerParenthesis : bool = True, pWeight : float = 0.5, considerRemoved : bool = True, removedWeight : float = 0.5, includeLabels : bool = False, includeThreshold : bool = False, threshold : float = 1.1):

        if considerParenthesis:
            if len(labels) != len(parenthesis):
                raise ValueError("labels and parenthesis must be the same length")

        if includeThreshold:
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("threshold must be 0.0 - 1.0")

            Similar = []
            n = len(labels)
            # iterate through each pair (i,j) and calculate their similarity via model.
            for i in range(n):

                i_terms = labels[i]
                i_paren = parenthesis[i]

                for j in range(i+1, n):

                    j_terms = labels[j]
                    j_paren = parenthesis[j]

                    simVal = self.SimWithParenthesis(model, i_terms, i_paren, j_terms, j_paren, pWeight, considerRemoved, removedWeight)

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
            # we dont care about parenthesis:
            Similar = []
            n = len(labels)
            # iterate through each pair (i,j) and calculate their similarity via model.
            for i in range(n):
                i_terms = labels[i]

                for j in range(i+1, n):
                    j_terms = labels[j]

                    simVal = self.Sim(model, i_terms, i_terms, considerRemoved, removedWeight)

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

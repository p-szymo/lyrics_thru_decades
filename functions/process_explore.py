# statistical packages
from scipy.stats import ttest_ind

# string processing
import re
import string

# nlp packages
import nltk
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from textblob import TextBlob as tb
import pronouncing



# count number of words in a text
def word_counter(song):

    '''
    Function to count the number of words in a list of strings.


    Input
    -----
    song : str
        Song with words to count.


    Output
    ------
    word_count : int
        Total number of words in song.

    '''

    # create list of words
    words = song.split()

    # add up the word counts of each line
    word_count = len(words)

    return word_count


def unique_word_counter(lines):

    '''
    Function to count the number of unique words in a list of
    strings.


    Input
    -----
    lines : list (str)
        List of strings to count.


    Output
    ------
    num_unique : int
        Total number of unique words across all strings.

    '''

    # lowercase all words, remove punctuation and en dashes
    process_lines = [line.lower().replace('—', ' ').
                     translate(str.maketrans('', '', string.punctuation))
                     for line in lines]

    # list of all words
    words = [word for line in process_lines
             for word in line.split()]

    # number of unique words
    num_unique = len(set(words))

    return num_unique


# perform multiple two-way t-tests
def two_way_tests(series_list, names):

    '''
    Function to run a two-sided t-test on each combination within
    a list of Pandas Series values.

    Returns a dictionary with the indices of the tested series as
    the keys and the test results as the values.


    Input
    -----
    series_list : list (Pandas Series)
        A list in which each item is a Pandas Series of a
        continuous variable within a subsection of your data.

        Example Series format :
        ```df['*continuous_var'][df['*subsection'] == '*category']```


    Output
    ------
    compare_dict : dict
        Key : Tuple of indices of compared items from the input
            list.
        Value : Ttest_indResult() object with t-statistic and
            p-value.

    '''

    # instantiate empty dictionary
    compare_dict = {}

    # loop through indices
    for i in range(len(series_list)):
        # count represents index of compared item
        count = i+1
        # until it reaches the last item
        while count < len(series_list):
            # compare each combination
            compare_dict.update({(names[i], names[count]): ttest_ind(
                                                series_list[i],
                                                series_list[count]
                                            )}
                                )
            count += 1

    # return all comparisons
    return compare_dict


# count number of end rhymes in a text
def end_rhyme_counter(lines):

    '''
    Function to count the instances of rhymes that occur among
    the last words in lines (end rhymes).


    Input
    -----
    lines : list (str)
        List of strings to compare.


    Output
    ------
    sum(rhyme_counts) : int
        The number of end rhymes.

    '''

    # instantiate an empty dictionary
    rhyme_dict = {}

    # make a list of words at the end of the line
    end_words = [line.split()[-1].translate
                 (str.maketrans('', '', string.punctuation))
                 for line in lines]

    # loop to build the dictionary
    for word in end_words:
        for i in range(len(end_words)):

            # check if a word rhymes with another word in the list
            if end_words[i] in pronouncing.rhymes(word):

                # check if word is already a key in the dictionary
                if word not in rhyme_dict:

                    # or if its rhyming word is already a key
                    # in the dictionary
                    if end_words[i] not in rhyme_dict:

                        # if neither is, create the word as key and
                        # it's rhyme as a value (in a list)
                        rhyme_dict[word] = [end_words[i]]

                else:
                    # if word is already a key, append its rhyme to
                    # its value
                    rhyme_dict[word].append(end_words[i])

    # count up the amount of (unique) rhymes per word
    rhyme_counts = [len(rhyme) for rhyme in rhyme_dict.values()]

    return sum(rhyme_counts)


# count total syllables in text
def syllable_counter(lines):

    '''
    Function to count all syllables in a list of strings.

    NOTE: This does not factor in multi-syllabic digits,
    times (i.e. 1:03), and most likely other non-"word" words.


    Input
    -----
    lines : list (str)
        List of strings to count.


    Output
    ------
    sum(total) : int
        Total number of syllables in the input list.



    [Modified from Allison Parrish's example in the documention
     for her library, pronouncing]:
    https://pronouncing.readthedocs.io/en/latest/tutorial.html

    '''
    # create empty list
    total = []

    # loop over list
    for line in lines:

        # turn each word into a string of its phonemes
        # if else statement ensures that each word is counted with
        # at least one syllable, even if that word is not in the
        # pronouncing library's dictionary (using phoneme for 'I'
        # as a placeholder for single syllable)
        phonemes = [pronouncing.phones_for_word(word)[0]
                    if pronouncing.phones_for_word(word)
                    else 'AY1' for word in line.split()]

        # count the syllables in each string and add the total
        # syllables per line to the total list
        total.append(sum([pronouncing.syllable_count(phoneme)
                          for phoneme in phonemes]))

    # return the total number of syllables
    return sum(total)


# apply text cleaning techniques
def clean_text(text, stop_words):

    '''
    Function to make text lowercase, tokenize words and words with
    apostrophes, convert contractions to full words, lemmatize by
    POS tag, and remove stop words and words shorter than 3
    characters.


    Input
    -----
    text : str
        Text to be cleaned.

    stop_words : list (str)
        List of words to remove from the text.


    Output
    ------
    text : str
        Lowercase, lemmatized text without contractions, stop words,
        and one- to two-letter words.

    '''

    # make text lowercase and convert some punctuation
    text = text.lower().replace("’", "'")

    # remove punctuation other than apostrophes
    text = text.translate(str.maketrans(
        '', '', string.punctuation.replace("'", "")))

    # initial tokenization to remove non-words
    tokenizer = RegexpTokenizer("([a-z]+(?:'[a-z]+)?)")
    words = tokenizer.tokenize(text)

    # convert contractions
    contractions = load_dict_contractions()
    words = [contractions[word] if word in contractions else
             word for word in words]

    # stringify and remove leftover apostrophes
    text = ' '.join(words)
    text = text.replace("'", "")

    # remove stop words, lemmatize using POS tags,
    # and remove two-letter words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word))
             for word in nltk.word_tokenize(text)
             if word not in stop_words]

    # removing any words that got lemmatized into a stop word
    words = [word for word in words if word not in stop_words]

    # removing words less than 3 characters
    words = [word for word in words if len(word) > 2]

    # rejoin into a string
    text = ' '.join(words)

    return text


# obtain POS tags
def get_wordnet_pos(word):

    '''
    Function to map part-of-speech tag to first character
    lemmatize() accepts.


    Input
    -----
    word : str
        Word to tag.


    Output
    ------
    tag : wordnet object
        POS tag in the necessary format for WordNetLemmatizer().



    [Code borrowed from]:
    https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

    '''

    # get primary tag
    tag = nltk.pos_tag([word])[0][1][0].upper()

    # proper format conversion dictionary
    tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}

    # tag, if known; use noun if unknown
    return tag_dict.get(tag, wordnet.NOUN)
    

# contractions conversions
def load_dict_contractions():

    '''
    Dictionary of contractions as keys and their expanded words
    as values.


    [Code modified from]:
    https://stackoverflow.com/questions/19790188/expanding-english-\
    language-contractions-in-python

    '''

    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "cuz": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "d'you": "do you",
        "e'er": "ever",
        "em": "them",
        "'em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i'm'a": "i am about to",
        "i'm'o": "i am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "i've": "i have",
        "kinda": "kind of",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "needn't've": "need not have",
        "ne'er": "never",
        "o'": "of",
        "o'clock": "of the clock",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shalln't": "shall not",
        "shan't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "so's": "so as",
        "so've": "so have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "to've": "to have",
        "'tis": "it is",
        "tis": "it is",
        "'twas": "it was",
        "twas": "it was",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "will've": "will have",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why've": "why have",
        "why's": "why is",
        "won't": "will not",
        "won't've": "will not have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "would've": "would have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        }
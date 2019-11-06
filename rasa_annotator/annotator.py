import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd
import json

# Path must be read from a location given by the folder where the un-annotated sentences are
drink_sentences = "sentences_drink.txt"
sentences_df = pd.read_csv(drink_sentences, header=0)


def listing(sentence):
    """Takes the sentence in a Row data frame and put it inside a list"""
    return [sentence]


def tokenize_sentence(sentence):
    """Takes the output from Listing within the Dataframe and tokenizes it"""
    tk = word_tokenize(sentence[0])
    return tk


def tagger(sentence, entity, label):
    """For every entity given,find if the entity is contained in the sentence at turn

    if the Entity is present in the sentence, take such entity and convert it to [entity] along with the label provided

    output: lorem ipsum lorem ipsum [entity](label)
    """
    detkn = TreebankWordDetokenizer()

    try:

        if entity in sentence:

            out = []

            index_selector = sentence.index(entity)
            for counter, word in enumerate(sentence, 0):

                if counter != index_selector:

                    out.append(word)
                elif counter == index_selector:
                    matcher = '[' + word + ']'
                    out.append(matcher)
                    if sentence[-1] != word:
                        # [Dám, si, kávu]
                        out.append(sentence[-1])
                    index_finder = out.index(matcher)
                    out.insert(index_finder + 1, (label))
                    sentence_out = detkn.detokenize(out)

            return sentence_out

        else:

            return sentence

    except:
        return sentence


def untokenizer(sentence):
    """ Takes the non-annotated sentences and convert them to string"""
    if isinstance(sentence, list):
        detkn = TreebankWordDetokenizer()
        sentence = detkn.detokenize(sentence)
        return sentence
    else:
        return sentence


""" Pipeline to Annotate the sentences

    1. Listing
    2. Tokenize
    3. Annotate given a list of entities
    4. Untokenize non- annotated sentences """

# 1. Listing
sentences_df["'intent':'order_drink'"] = sentences_df["'intent':'order_drink'"].apply(listing)

# 2. Tokenize
sentences_df["'intent':'order_drink'"] = sentences_df["'intent':'order_drink'"].apply(tokenize_sentence)

# 2.1. Hand Picked entities
entities = ['pivo', 'kávu', 'kolu', 'kozel']

# 2.2. Hand Picked label, this must be dynamic
label = '(drink)'

# 3. Annotation
for entity in entities:
    sentences_df["'intent':'order_drink'"] = sentences_df["'intent':'order_drink'"].apply(tagger, entity=entity,
                                                                                          label=label)

# 4. Untokenize
sentences_df["'intent':'order_drink'"] = sentences_df["'intent':'order_drink'"].apply(untokenizer)

""" Training Data consolidator 

    1. Takes the  Annotated/Non-annotated sentences and puts them in a dictionary following the structure:
        {"intent": "name of intent",
         "text": sentences_df["'intent':'order_drink'"][i],
         "index": i
        })
        
    2. Converts the Dict data structure to Json and stores it as training_data_json
     """

dataset_nn = []
for i, sentence in enumerate(sentences_df["'intent':'order_drink'"], 0):
    dataset_nn.append(
         {"intent": "order_drink",
         "text": sentences_df["'intent':'order_drink'"][i],
         "index": i
        })

with open('training_data.json', 'w', encoding='utf8') as json_file:
    json.dump(dataset_nn, json_file, ensure_ascii=False, indent=4)
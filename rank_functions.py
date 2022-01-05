import builtins
import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))

def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens

def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query (unique terms).
    Each entry within this vector represents a tfidf score for each term in the query.

    For calculation of IDF, we use log with base 10, and tf will be normalized based on the length of the query.
    Parameters:
    -----------
    query_to_search: list of tokens (str).
    index: inverted index loaded from the corresponding files.
    """

    epsilon = .0000001
    unique_query_terms = np.unique(query_to_search)
    Q = np.zeros(len(unique_query_terms))
    counter = Counter(query_to_search)
    for token in unique_query_terms:
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] /len(query_to_search) # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL))/(df + epsilon), 10)  # smoothing
            try:
                ind = unique_query_terms.index(token)
                Q[ind] = tf*idf
            except:
                pass
    return Q

def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.
    This function will go through every token in query_to_search,
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str).
    index: inverted index loaded from the corresponding files.
    words,pls: generator for working with posting.
    -----------
    Returns: dictionary of candidates. In the following format: key: pair (doc_id,term), value: tfidf score.
    """
    candidates = {}
    N = len(index.DL)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            # update - Removed str function in DL[doc_id] | normlized_tfidf = [(doc_id,(freq/index.DL[str(doc_id)])*math.log(N/index.df[term],10)) for doc_id, freq in list_of_doc]
            normlized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq in list_of_doc]
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
    return candidates

def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query.
    Columns will be the unique terms in the query.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str)
    index: inverted index loaded from the corresponding files.
    words,pls: generator for working with posting.
    """

    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words, pls)
    # Dictionary of unique candidates In the following format: key: pair (doc_id,term), value: tfidf score.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    # unique terms of the query
    unique_query_terms = np.unique(query_to_search)
    # DataFrame of zeros , unique_candidates (row) * unique_query_terms (col)
    D = np.zeros((len(unique_candidates), len(unique_query_terms)))
    D = pd.DataFrame(D)
    # set index for rows and columns
    D.index = unique_candidates
    D.columns = unique_query_terms

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D

def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """

    dic = {}
    mat = np.dot(Q, np.transpose(D)) / (np.linalg.norm(Q) * (np.linalg.norm(D, ord=None, axis=1)))
    dic = {}
    ln = mat.shape[0]
    for i in range(ln):
        dic[D.index[i]] = mat[i]
    return dic
    # df_cosine_similarity = defaultdict(int)
    # q_len = len(Q)
    # for index, row in D.iterrows():
    #   di_len = len(row)
    #   numerator = np.dot(row,Q)
    #   denominator = q_len*di_len
    #   df_cosine_similarity[index] = (numerator/denominator)
    # return df_cosine_similarity


def get_top_n(sim_dict, N=100):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                  reverse=True)[:N]


def get_topN_score_for_query(query_to_search, index, words, pls, N=100):
    """
    Generate a dictionary that gathers for a query its topN score.

    query_to_search: list of tokens

    Returns: a ranked (sorted) list of pairs (doc_id, score) in the length of N.
    """
    vec_q = generate_query_tfidf_vector(query_to_search, index)
    mat_tfidf = generate_document_tfidf_matrix(query_to_search, index, words, pls)
    sim_dict = cosine_similarity(mat_tfidf, vec_q)
    return get_top_n(sim_dict, N)


def get_candidate_documents(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    list of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = []
    for term in np.unique(query_to_search):
        if term in words:
            current_list = (pls[words.index(term)])
            candidates += current_list
    return np.unique(candidates)


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    merged_dict = {}

    for query_id in title_scores:
        title_values = title_scores[query_id]
        body_values = body_scores[query_id]

        doc_to_score = defaultdict(int)

        for doc, score in title_values:
            doc_to_score[doc] += title_weight * score

        for doc, score in body_values:
            doc_to_score[doc] += text_weight * score

        merged_dict[query_id] = sorted([(doc_id, score) for doc_id, score in doc_to_score.items()], key=lambda x: x[1],
                                       reverse=True)[:N]

    return merged_dict

def get_documents_by_content(query_to_search, index, words, pls):
    """
    Returns ALL (not just top 100) documents that contain A QUERY WORD
    IN THE TITLE of the article, ordered in descending order of the NUMBER OF
    QUERY WORDS that appear in the title.
    return : list of (doc id, number of words) , sort in descending order
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            for doc_item in list_of_doc:
                candidates[doc_item[0]] = candidates.get(doc_item[0], 0) + doc_item[1]
    return sorted(candidates.items(), key=lambda item: item[1], reverse=True)


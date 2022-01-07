import math
import numpy as np
import rank_functions as rf

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM_25_from_index:
    """
    Calculates Best Match by BM25 for a given query.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75
    index: inverted index
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.idf = None

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        list_of_tokens: list of token representing the query.
        Returns: dictionary of idf scores. As follows: key: term, value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_candidate_documents(query_to_search, words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query.
        Parameters:
        -----------
        query_to_search: list of tokens (str).
        index: inverted index loaded from the corresponding files.
        words,pls: generator for working with posting.
        Returns:
        -----------
        list of candidates ids
        """
        candidates = []
        unique_q = np.unique(query_to_search)
        for term in unique_q:
            if term in words:
                current_list = (pls[words.index(term)])
                for item in current_list:
                    candidates.append(item[0])
        return np.unique(candidates)

    def search(self, query, N=100, query_words=None, query_pls=None):
        """
        This function calculate the bm25 score for given query and document.
        We check only documents which are 'candidates' for a given query.

        query = list of token representing the query.

        This function return a dictionary of scores as the following: key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.
        """
        candidates = self.get_candidate_documents(query, query_words, query_pls)
        self.idf = self.calc_idf(query)
        doc_and_scores = {}
        for doc_id in candidates:
            doc_and_scores[doc_id] = self._score(query, doc_id, query_words, query_pls)
        res = rf.get_top_n(doc_and_scores, N)
        return res

    def _score(self, query, doc_id, query_words=None, query_pls=None):
        """
        Calculate the bm25 score for a given query and document.
        query: list of token representing the query.
        doc_id: integer, document id.
        Returns score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.index.DL[doc_id]
        for term in query:
            if term in self.index.term_total.keys():
                term_frequencies = dict(query_pls[query_words.index(term)])
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.index.avgDl)
                    score += (numerator / denominator)
        return score
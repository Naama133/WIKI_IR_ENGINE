import os
import pickle
from collections import defaultdict
from pathlib import Path
from flask import Flask, request, jsonify
from google.cloud import storage
import inverted_index_gcp
import rank_functions as rf
import BM_25_from_index as bm25

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# access GCP project, and download the indexes from the storage bucket
bucket_name = 'project_ir_test'
client = storage.Client('elated-chassis-334219')
bucket = client.bucket(bucket_name)

# download pickle file from stoage bucket
def get_content_from_storage(bucket, file_name):
    blob = storage.Blob(f'{file_name}', bucket)
    with open(f'./{file_name}', "wb") as file_obj:
        blob.download_to_file(file_obj)
    with open(f'./{file_name}', 'rb') as f:
        return pickle.load(f)

print("start reading page rank")
# Download page rank calculations from storage and save it into the doc_id_2_page_rank variables
doc_id_2_page_rank = get_content_from_storage(bucket, "pr_pagerank2dict.pckl")
print("start reading page view")
# Download "page view - August 2021" file and save it into the wid2pv variables
wid2pv = get_content_from_storage(bucket, "pageviews-202108-user.pkl")

# download index file and save it into the indexes variables
def get_index_from_storage(bucket, storage_path, index_name):
    blob = storage.Blob(f'postings_gcp/{storage_path}/{index_name}.pkl', bucket)

    with open(f'./{index_name}.pkl', "wb") as file_obj:
        blob.download_to_file(file_obj)

    return inverted_index_gcp.InvertedIndex.read_index("./", index_name)

#download bins files
def get_bins_from_storage(bucket_name, storage_path):

    os.makedirs(f'./postings_gcp/{storage_path}', exist_ok=True)
    blobs = client.list_blobs(bucket_name, prefix=f'postings_gcp/{storage_path}')

    for blob in blobs:
        if blob.name.endswith('.bin'):
            with open(f'./{blob.name}', "wb") as file_obj:
                blob.download_to_file(file_obj)


def get_posting_gen(index, bin_directory, query):
    """
    Return the generator working with posting list.
    Parameters: index: inverted index
    ----------
    """
    words, pls = zip(*index.posting_lists_iter(bin_directory, query))
    return words, pls

# Create 3 inverted indexes of body, title and anchor text
print("starting reading index_body")
storage_path_body = "index_body"
body_index = get_index_from_storage(bucket, storage_path_body, 'index_body')
get_bins_from_storage(bucket_name, storage_path_body)
print("starting reading index_title")
storage_path_title = "index_title"
title_index = get_index_from_storage(bucket, storage_path_title, 'index_title')
get_bins_from_storage(bucket_name, storage_path_title)
print("starting reading index_anchor_text")
storage_path_anchor_text = "index_anchor_text"
anchor_text_index = get_index_from_storage(bucket, storage_path_anchor_text, 'index_anchor_text')
get_bins_from_storage(bucket_name, storage_path_anchor_text)


def search_anchor_func(query):
    # words & posting lists of each index
    words_anchor_text, pls_anchor_text = get_posting_gen(anchor_text_index, 'postings_gcp/index_anchor_text', query)
    sorted_docs_list = rf.get_documents_by_content(query, anchor_text_index, words_anchor_text, pls_anchor_text)
    return dict(sorted_docs_list[:100])

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    #calculate BM25 on the body index
    tokenized_query = rf.tokenize(query)
    # words & posting lists of each index
    if len(tokenized_query) == 0:
        return jsonify(res)
    words_body, pls_body = get_posting_gen(body_index, 'postings_gcp/index_body', tokenized_query)
    bm25_body = bm25.BM_25_from_index(body_index)
    bm25_scores = bm25_body.search(tokenized_query, 200, words_body, pls_body)
    anchor_values = search_anchor_func(tokenized_query)
    anchor_weight = 3
    doc_to_score = defaultdict(int)

    for doc_id, bm25score in bm25_scores:
        pageview = wid2pv.get(doc_id, 1)
        doc_to_score[doc_id] += (2 * bm25score * pageview) / (bm25score + pageview)
    for doc, score in anchor_values.items():
        doc_to_score[doc] *= anchor_weight * score

    doc_to_score = sorted([(doc_id, score) for doc_id, score in doc_to_score.items()], key=lambda x: x[1], reverse=True)[:100]

    for doc_id, score in doc_to_score:
        res.append((int(doc_id), title_index.doc_id_to_title.get(doc_id, "")))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = rf.tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    # words & posting lists of each index
    words_body, pls_body = get_posting_gen(body_index, 'postings_gcp/index_body', tokenized_query)
    docs_scores = rf.get_topN_score_for_query(tokenized_query, body_index, words_body, pls_body) # A ranked (sorted) list of pairs (doc_id, score) in the length of N
    for item in docs_scores:
        res.append((int(item[0]), title_index.doc_id_to_title.get(item[0], "")))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = rf.tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    # words & posting lists of each index
    words_title, pls_title = get_posting_gen(title_index, 'postings_gcp/index_title', tokenized_query)
    sorted_docs_list = rf.get_documents_by_content(tokenized_query, title_index, words_title, pls_title)
    for item in sorted_docs_list: ## naama
        res.append((int(item[0]), title_index.doc_id_to_title.get(item[0], "")))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = rf.tokenize(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    # words & posting lists of each index
    words_anchor_text, pls_anchor_text = get_posting_gen(anchor_text_index, 'postings_gcp/index_anchor_text', tokenized_query)
    sorted_docs_list = rf.get_documents_by_content(tokenized_query, anchor_text_index, words_anchor_text, pls_anchor_text)
    for item in sorted_docs_list:
        res.append((int(item[0]), title_index.doc_id_to_title.get(item[0], "")))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        res.append(doc_id_2_page_rank.get(doc_id, 0.0))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    '''
    Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        res.append(wid2pv.get(doc_id, 0))
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

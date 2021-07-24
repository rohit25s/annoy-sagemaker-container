# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import io
import sys
import signal
import traceback
import csv
import json
import flask
import datetime

import pandas as pd

from annoy import AnnoyIndex

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

with open(os.path.join(model_path, 'training-params-map.pkl'), 'rb') as inp:
        savedParams = pickle.load(inp)

dim = int(savedParams.get('feature_dim', 80))
metric = savedParams.get('metric', 'angular')
nns = int(savedParams.get('nns', 100))
search_k = int(savedParams.get('search_k', -1))
    
i_map = None
index = None 

if i_map == None:
    with open(os.path.join(model_path, 'annoy-index-map.pkl'), 'rb') as inp:
        i_map = pickle.load(inp)
        
print("dim: " + str(dim) +", metric: " + metric)
if index == None:
    index = AnnoyIndex(dim, metric)
    index_path = os.path.join(model_path, "index.ann")
    index.load(index_path)    

# A singleton for holding the index. This simply loads the index and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    
    @classmethod
    def load_i_map(cls):
        if cls.i_map == None:
            with open(os.path.join(model_path, 'annoy-index-map.pkl'), 'rb') as inp:
                cls.i_map = pickle.load(inp)
        return cls.i_map

    @classmethod
    def load_index(cls, dim, metric):
        print("dim: " + str(dim) +", metric: " + metric)
        if cls.index == None:
            cls.index = AnnoyIndex(dim, metric)
            index_path = os.path.join(model_path, "index.ann")
            cls.index.load(index_path)
        return cls.index

    @classmethod
    def getnns(cls, i, k, nns):
        knn = cls.index.get_nns_by_item(i, nns, search_k=k)
        knn_to_id = list(map(lambda x: cls.i_map[x], knn))
        top_k=[cls.i_map[i]]
        top_k.append(knn_to_id)
        return top_k

    @classmethod
    def getnnsbyvector(cls, row, k, nns):
        item = int(row[1][0])
        vec = row[1][1]
        vector = [float(item) for item in vec[1:-1].split()]
        print("type:"+str(type(vector)))
        print("length:"+str(len(vector)))
        knn = index.get_nns_by_vector(vector, nns, search_k=k)
        knn_to_id = list(map(lambda x: i_map[x], knn))
        top_k=[item]
        top_k.append(knn_to_id)
        return top_k

    @classmethod
    def predict(cls, data, nns, search_k):

        start_time = datetime.datetime.now()
        predictions = []
        
        for row in data.iterrows():
            predictions.append(cls.getnnsbyvector(row, search_k, nns))
        
        query_time = datetime.datetime.now() - start_time
        print("time for " + str(len(data)) + " reads = " + str(query_time))

        return predictions

# The flask app for serving predictions
app = flask.Flask(__name__)



@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if i_map length is same as indexed items."""
    result = "i_map length:" + str(len(i_map)) + ", n_items: " + str(index.get_n_items())
    index_path = os.path.join(model_path, "index.ann")
    if len(i_map) == index.get_n_items() and os.path.exists(index_path):
        status = 200 
    else:
        status = 404
    return flask.Response(response=result, status=status, mimetype='text/plain')

@app.route('/invocations', methods=['POST'])
def transformation():
    data = None

    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None, sep =",")
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))
    print(str(data))
    
    predictions = ScoringService.predict(data, nns, search_k)
    print("predictions done")

    # Convert from numpy back to CSV
    out = io.StringIO()
    csvWriter = csv.writer(out,delimiter=',')
    csvWriter.writerows(predictions)

    result = out.getvalue()

    print("result done")

    return flask.Response(response=result, status=200, mimetype='text/csv')


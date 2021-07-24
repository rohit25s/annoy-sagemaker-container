# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import sys
import pickle
import datetime

from joblib import Parallel, delayed
import multiprocessing

import pandas as pd

from annoy import AnnoyIndex

prefix = '/opt/ml/'
input_path = prefix + 'input/data'
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
test_channel='test'
test_path=os.path.join(input_path, test_channel)

class Ann:
    model = None
    i_map = None
    index = None
    key_list = None
    val_list = None

    def __init__(self):
        pass

    def load_list(self):
        key_list=list(self.i_map.keys())
        val_list=list(self.i_map.values())
        return None

    #since annoy uses iter values for indexing, need to have an i_map to correlate between iter and actual id
    def load_i_map(self):
        if self.i_map == None:
            with open(os.path.join(model_path, 'annoy-index-map.pkl'), 'rb') as inp:
                self.i_map = pickle.load(inp)
        return self.i_map

    def load_index(self, dim, metric):
        if self.index == None:
            self.index = AnnoyIndex(dim, metric)
        index_path = os.path.join(model_path, "index.ann")
        self.index.load(index_path)
        return self.index

    def getnns(self, i, k, nns, include_distances):
        knn = self.index.get_nns_by_item(i, nns, search_k=k, include_distances=include_distances)
        if include_distances:
            dist = knn[1]
            knn_to_id = list(map(lambda x: self.i_map[x], knn[0]))
        else:
            dist = []
            knn_to_id = list(map(lambda x: self.i_map[x], knn))
        top_k=[self.i_map[i]]
        top_k.append(knn_to_id)
        top_k.append(dist)
        return top_k

    def getnnsbyvector(self, row, k, nns, include_distances):
        knn = self.index.get_nns_by_vector(row[1][1], nns, search_k=k, include_distances=include_distances)
        if include_distances:
            dist = knn[1]
            knn_to_id = list(map(lambda x: self.i_map[x], knn[0]))
        else:
            dist = []
            knn_to_id = list(map(lambda x: self.i_map[x], knn))
        top_k=[row[1][0]]
        top_k.append(knn_to_id)
        top_k.append(dist)
        return top_k

    def getnnsbyitem(self, row, k, nns, include_distances):
        ind = self.val_list.index(row[1][0])
        key = self.key_list[ind]
        knn = self.index.get_nns_by_item(key, nns, search_k=k, include_distances=include_distances)
        if include_distances:
            dist = knn[1]
            knn_to_id = list(map(lambda x: self.i_map[x], knn[0]))
        else:
            dist = []
            knn_to_id = list(map(lambda x: self.i_map[x], knn))
        top_k=[row[1][0]]
        top_k.append(knn_to_id)
        top_k.append(dist)
        return top_k

    def predict_all(self, nns, dim, metric, search_k, include_distances):

        predictions = []

        start_time = datetime.datetime.now()


        i_map = self.load_i_map()
        #self.load_list()
        self.load_index(dim, metric)
        print("loading time = " + str(datetime.datetime.now() - start_time))

        cpu_count = multiprocessing.cpu_count()

        predictions = Parallel(n_jobs=cpu_count, require="sharedmem")(
            delayed(self.getnns)(i, search_k, nns, include_distances) for i in range(self.index.get_n_items())
        )

        query_time = datetime.datetime.now() - start_time
        print("time for " + str(self.index.get_n_items()) + " reads = " + str(query_time))

        return predictions


    def predict_for_vector(self, df, nns, dim, metric, search_by_vector, search_k, include_distances):

        predictions = []

        start_time = datetime.datetime.now()

        i_map = self.load_i_map()
        self.load_list()
        self.load_index(dim, metric)

        cpu_count = multiprocessing.cpu_count()

        if str(search_by_vector) == 'True':
            predictions = Parallel(n_jobs=cpu_count, require="sharedmem")(
                delayed(self.getnnsbyvector)(row, search_k, nns, include_distances) for row in df.iterrows()
            )
        else:
            predictions = Parallel(n_jobs=cpu_count, require="sharedmem")(
                delayed(self.getnnsbyitem)(row, search_k, nns, include_distances) for row in df.iterrows()
            )

        query_time = datetime.datetime.now() - start_time
        print("time for " + str(len(df)) + " reads = " + str(query_time))

        return predictions


    def transformation(self):
        data = None

        with open(os.path.join(model_path, 'training-params-map.pkl'), 'rb') as inp:
            savedParams = pickle.load(inp)

        #load hyperparameters
        dim = int(savedParams.get('feature_dim', 80))
        metric = savedParams.get('metric', 'angular')
        nns = int(savedParams.get('nns', 100))
        search_by_vector = savedParams.get('search_by_vector', False)
        search_k = int(savedParams.get('search_k', -1))
        knn_all = savedParams.get('knn_all', True)
        output_path = savedParams.get('output_path', None)
        include_distances = savedParams.get('include_distances', True)

        if str(include_distances) == 'False':
            include_dist = False
        else:
            include_dist = True

        #if test dataset is different or part of training dataset, prediction is done using feature vector
        if str(knn_all) == 'False':
            if not os.path.exists(test_path):
                print("No test data found, exiting!!!")
                sys.exit(0)
            files = [f for f in os.listdir(test_path) if f.endswith("parquet")]

            print("{} parquet files found. Beginning reading...".format(len(files)), end="")
            start = datetime.datetime.now()

            df_list_test = [pd.read_parquet(os.path.join(test_path, f)) for f in files]
            df = pd.concat(df_list_test, ignore_index=True)

            end = datetime.datetime.now()
            print(" Finished. Took {}".format(end-start))

            print("dataframe data types: " + str(df.dtypes))
            print("dataframe size: " + str(len(df)))

            predictions = self.predict_for_vector(df, nns, dim, metric, search_by_vector, search_k, include_dist)

        else:
            predictions = self.predict_all(nns, dim, metric, search_k, include_dist)

        print("predictions done")
        time = datetime.datetime.now()
        final_path = output_path + "/final_output_" + str(time) + ".parquet"
        df = pd.DataFrame(predictions, columns = ['id', 'top_k', 'distances'])
        df.to_parquet(final_path)
        print("written to s3, path:" + final_path)

        return None

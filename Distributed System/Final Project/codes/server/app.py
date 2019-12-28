from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS
import time

import pyspark
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF


app = Flask(__name__)
# cors = CORS(app, resources={r"/search": {"origins": "*"}})


@app.route('/')
def mainPage():
    # return 'test for mainPage'
    return render_template("./search.html")


@app.route('/search', methods=['GET', 'POST'])
def search():

    # #test for json pass
    # result = sparkCal()
    # return jsonify(result)

    if request.method == 'POST':
        keyWord = request.form['keyWord']
        # # test for vue json
        # result = sparkCal(keyWord)
        # return jsonify(result)
        start_time = time.time()
        result = sparkCal(keyWord)
        end_time = time.time()
        nums = len(result)
        search_result = []
        for item in result:
            search_result.append({'title': item[0], 'relation': item[1]})
        time = end_time - start_time

        return render_template("result.html", search_result=search_result, nums=nums, keyWord=keyWord, time=time)


def sparkCal(term):
    
    def topDocsForTerm(term):
        try:
            index = tf.indexOf(term)
#             print("index of term", index)
            term_row = V[index]
        except:
            print("Term doesn't exist")
            return

        cosine_sim = np.dot(U, np.dot(S, term_row))
        indeces = np.argsort(cosine_sim).tolist()
        indeces.reverse()

        return list(zip(titles[indeces[:10]], cosine_sim[indeces]))

    conf = (SparkConf().set("spark.local.ip", "10.192.7.116"))
    sc = SparkContext(conf=conf)

    US_normalized = np.load("US_normalized.npy")
    V = np.load("V.npy")
    S = np.load("S.npy")
    U = np.load("U.npy")
    titles = np.load("titles.npy")
    tf = HashingTF(50000)
    res = topDocsForTerm(term)

    sc.stop()

    # # test for json pass
    # return 'test for json'
    return res


if __name__ == "__main__":
    app.run()

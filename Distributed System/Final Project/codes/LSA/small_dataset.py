# coding:utf-8
import pyspark
from pyspark import SparkConf, SparkContext

# 设置spark
# sc.stop()
conf = SparkConf().setAppName('Small').setMaster('local')   # 小数据集用单机运行即可
sc = SparkContext(conf=conf)

log = sc.textFile("small100000")    # 读取小数据集

#####################################################
################ Step 1 清理数据 ####################
#####################################################

# 识别出每篇文章的title
import re
def id(line):
    try:
        title = re.findall("<title>(.+?)</title>", line, flags=0)   # 用正则法则定位每篇wiki的title
        if len(title)>0:    # 定位标题中的大写字母，在其前方加空格。（爬数据的时，有些标题的各个单词没有分开）
            l=list("".join(title))
            l1 = []
            for i in range(len(l)):
                if l[i].isupper() and (i>0):
                    l1.append(" ")
                l1.append(l[i])
            return [1,''.join(l1)]  # 返回 [0, title]，0表示title不为空
        else:
            return [0,None]
    except:
        return [0,None]

log = log.map(lambda x:[x,id(x)[0],id(x)[1]]) # 形如 [content, 0/1, title]

# 加载nltk和gensim语料库，对wiki文章内容做预处理
# 处理方法参考：https://github.com/dbaikova/Wikipedia_LSA/blob/master/WikipediaAnalysisWithLSA.ipynb
import gensim
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')    # 若无nltk的stopwords语料库，则先下载
stop_words = stopwords.words("english")
def short(log):
    # log = log.lower()
    log = str(log)
    log = gensim.parsing.preprocessing.strip_tags(log)    # 删除html的标签
    log = gensim.parsing.preprocessing.strip_punctuation(log) # 删除标点
    log = gensim.parsing.preprocessing.strip_numeric(log) # 删除数字
    log = gensim.parsing.preprocessing.remove_stopwords(log)  # 删除形如is, the, a等无用的词
    stemmer = gensim.parsing.porter.PorterStemmer()
    # log = stemmer.stem(log) # 提取词干
    log = gensim.parsing.preprocessing.strip_short(log, minsize=3)    # 删除过短的单词
    return log
data_clean = log.map(lambda x: [short(x[0]), x[1], x[2]]) # 形如 [content, 0/1, title]，content是处理过的文本内容

# 给每段内容标号，识别其属于第几篇文章
sum = 0.
def give_index(line):
    global sum
    if line[1] == 0:
        line[1] += sum
    else:
        line[1] += sum
        sum += 1
    return line
data_clean2 = data_clean.map(give_index) # 返回形如 [content, index, title]
data = data_clean2.filter(lambda x:len(x[0])>0).map(lambda x:(x[1],x[0],x[2])) # 清理content为空的行，并返回形如 [index, content, title]

# 整合每篇文章的所有内容
i_str = ''
i_num=""
title_before=''
title_now=''
def combine(line):
    global i_str
    global i_num
    global result
    global title_before
    global title_now
    if line[0] == i_num:
        # 相同文章的内容，则全局变量i_str连接上这部分内容
        i_str = ' '.join([i_str,line[1]])
        # 返回空值，以免扩大数据量
        return ''
    elif line[2]:
        title_before = title_now
        title_now = line[2]
        i_num = line[0]
        tmp = i_str
        i_str = line[1]
        if len(title_before):
            # 一篇文章结束，返回其index以及所有内容
            return (i_num - 1,tmp,title_before)
        else:
            return ''
    else:
        return ""
data = data.map(combine).filter(lambda x:len(x)>0) # 返回形如 [index, content, title]，完成数据预处理

#####################################################
################# Step 2 TF-IDF #####################
#####################################################

from pyspark.mllib.feature import HashingTF,IDF

# TF部分
tf = HashingTF(50000)   # 取50000维
vectors = data.map(lambda line:(line[0],line[2],tf.transform(line[1]))) # 形如 [index, title, tf]

# IDF部分
vec = vectors.map(lambda line: line[2]) # 只留下tf结果
idf = IDF()
idfmodel = idf.fit(vec)
tfIdfVectors = idfmodel.transform(vec) # 获得tf-idf结果，完成tf-idf步骤

#####################################################
################## Step 3 SVD #######################
#####################################################

# 进计算VD
from pyspark.mllib.linalg.distributed import RowMatrix
import numpy as np

tfIdf_matrix = RowMatrix(tfIdfVectors)   # 将算好的tf-idf结果转换成矩阵
svd = tfIdf_matrix.computeSVD(100, True)    # 计算svd并只保留特征值最大的前100个
u = svd.U
s = svd.s
v = svd.V

V = v.toArray()
S = np.diag(s.toArray())    # s是对角矩阵
SV = np.dot(V, S)
SV_normalized = normalize(SV, 'l2') # SV是最后有用的，‘l2'指的是使用(欧几里德)L2-范数
aux = u.rows.map(lambda row: row.toArray())
U = np.array(np.array(aux.collect()))
US = np.dot(U, S)
US_normalized = normalize(US, 'l2') # US也有用

#####################################################
############ Step 4 文档/文档相关矩阵 ###############
#####################################################

# 给定一篇文章，返回其相关的文章
def relevant_docs(input_title):
    try:
        index = np.where(titles == input_title)[0][0]   # 找出查询的这篇文章
    except IndexError:
        return 'No such document'   # 找不到该关键词
    cosine_sim = np.dot(US_normalized, US_normalized[index]) # 计算其他文章与它之间的相似度
    indeces = np.argsort(cosine_sim).tolist()   # 从小到大排列后，返回其index
    indeces.reverse()   # 改变顺序，使其从大到小排列，这样越靠前则相似度越高
    return list(zip(titles[indeces[:10]], cosine_sim[indeces])) # 返回前10个相关的文章，形如 [(title, 相关度), ...]

titles = data.map(lambda line: line[2]) # 该表仅保留title
titles = np.array(titles.collect()) # 转成np.array

# 查询相关文章,以Autism为例
# print(relevant_docs('Autism'))

#####################################################
################## Step 4 查询 ######################
#####################################################

# 给定关键词，返回查询结果
def topDocsForTerm(term):
    try:
        index = tf.indexOf(term)    # 找出关键词对应的index
        term_row = V[index] # 关键词对应的特征向量
    except:
        print("Term doesn't exist") # 找不到该关键词
        return
    cosine_sim = np.dot(U, np.dot(S, term_row)) # 还原该列SVD之前的结果，即返回该词在对每篇文章的相关度
    indeces = np.argsort(cosine_sim).tolist()   # 排序
    indeces.reverse()   # 降序排列
    return list(zip(titles[indeces[:10]], cosine_sim[indeces])) # 返回前十篇相关的文章，形如 [title, 相关度]

# # 进行测试
# print(topDocsForTerm('History'))
# print('----')
# print(topDocsForTerm('Geography'))
import numpy as np
import time
import nltk
import pickle
import confusionMatrix
import kgram
import difflib
import eval
from nltk.corpus import reuters
from collections import deque

def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def transformType(originWord, errorWord):
    if "-" in originWord:
        originWord = originWord.replace("-", "#")
    if "-" in errorWord:
        errorWord = errorWord.replace("-", "#")
    d = difflib.Differ()
    diff = "".join(list(d.compare(originWord, errorWord)))
    diff = "".join(diff.split(" "))
    if diff.count("+") == 1:
        addIndex = diff.find("+")
        if diff.count("-") == 1:
            deleteIndex = diff.find("-")
            if addIndex - deleteIndex == 2 or deleteIndex - addIndex == 2:
                return "sub", diff[deleteIndex + 1], diff[addIndex + 1]
            elif (addIndex - deleteIndex == 3 or deleteIndex - addIndex == 3) and diff[addIndex + 1] == diff[
                deleteIndex + 1]:
                return "trans", diff[deleteIndex - 1], diff[deleteIndex + 1]
            elif edit_distance(originWord,errorWord) == 1:
                return "sub", diff[deleteIndex + 1], diff[addIndex + 1]
            else:
                return "more", "", ""
        elif diff.count("-") == 0:
            return "ins", diff[addIndex - 1], diff[addIndex + 1]
        else:
            return "more", "", ""
    elif diff.count("-") == 1:
        deleteIndex = diff.find("-")
        return "del", diff[deleteIndex - 1], diff[deleteIndex + 1]
    else:
        return "more", "", ""


# initial Channel Matrix

try:
    data = []
    with open("Matrix.pk", "rb") as Matrixf:
        data = pickle.load(Matrixf)
    delMatrix, insMatrix, subMatrix, transMatrix = data[0], data[1], data[2], data[3]
except:
    delMatrix, insMatrix, subMatrix, transMatrix = confusionMatrix.initMatrix2()


# Channel Model Prob
def Pxw(originWord, errorWord, corpus, corpus_str,lamda):
    type, letter1, letter2 = transformType(originWord, errorWord)
    if type == "ins":
        # print("ins",originWord,errorWord,letter1,letter2)
        if (letter1+letter2+"|"+ letter1) in insMatrix and corpus_str.count(letter1):
            if letter1 == " ":
                if corpus_str.count(" " + letter2):

                    P = (insMatrix[(letter1+letter2+"|"+ letter1)] + 1) / (corpus_str.count(" " + letter2) + (len(corpus)*lamda))

                else:
                    P = 1 / (corpus_str.count(letter1) + (len(corpus)*lamda))
            else:
                # for example acress|acres e s -> es|e

                P = (insMatrix[(letter1+letter2+"|"+ letter1)] + lamda)/ (corpus_str.count(letter1) + len(corpus)*lamda)

        elif corpus_str.count(letter1) :
            P = 1 / (corpus_str.count(letter1) + len(corpus)*lamda)
        elif (letter1+letter2+"|"+ letter1) in insMatrix:
            P = (insMatrix[(letter1+letter2+"|"+ letter1)] + lamda)/len(corpus) *lamda
        else:
            P = 1 / len(corpus)*lamda
        return P
    if type == "del":
        # print("del", originWord, errorWord, letter1, letter2)
        # for example acress|actress c t -> c|ct
        if (letter1+"|"+ letter1+letter2) in delMatrix and corpus_str.count(letter1 + letter2):
            P = (delMatrix[(letter1+"|"+ letter1+letter2)] + lamda) / (corpus_str.count(letter1 + letter2) + len(corpus)*lamda)
        elif corpus_str.count(letter1 + letter2):
            P = 1 / (corpus_str.count(letter1 + letter2) + (len(corpus)*lamda))
        elif (letter1+"|"+ letter1+letter2) in delMatrix:
            P = (delMatrix[(letter1+"|"+ letter1+letter2)] + lamda)/(len(corpus)*lamda)
        else:
            P = 1 / (len(corpus)*lamda)
        return P
    if type == "trans":
        # print("trans", originWord, errorWord, letter1, letter2)
        if (letter2+letter1+"|"+ letter1+letter2) in transMatrix and corpus_str.count(letter1 + letter2):
            # for example caress acress c a ->ac|ca
            P = (transMatrix[(letter2+letter1+"|"+ letter1+letter2)] + lamda ) / (corpus_str.count(letter1 + letter2) + (len(corpus)*lamda))
        elif corpus_str.count(letter1 + letter2):
            P = 1 / (corpus_str.count(letter1 + letter2) + (len(corpus)*lamda))
        elif (letter2+letter1+"|"+ letter1+letter2) in transMatrix\
                : P = (transMatrix[(letter2+letter1+"|"+ letter1+letter2)] + lamda)/(len(corpus)*lamda)
        else:
            P = 1 / len(corpus) *lamda
        return P
    if type == "sub":
        # print("sub", originWord, errorWord, letter1, letter2)

        if (letter1+"|"+ letter2) in subMatrix and corpus_str.count(letter2):
            # for example acress|access r c -> r|c
            P = (subMatrix[(letter1+"|"+ letter2)] + 1) / (corpus_str.count(letter2) + (len(corpus)*lamda))
        elif corpus_str.count(letter2):
            P = 1 / (corpus_str.count(letter2) + (len(corpus)*lamda))
        elif (letter1+"|"+ letter2) in subMatrix:
            P = (subMatrix[(letter1+"|"+ letter2)] + lamda)/(len(corpus)*lamda)
        else:
            P = 1 / (len(corpus)*lamda)
        return P
    if type == "more" :
        if edit_distance(originWord,errorWord) == 1:
            print("more", originWord, errorWord, letter1, letter2)
        P = 1 / (len(corpus)*lamda)

    return P



try:
    data = []
    with open("Count.pk", "rb") as fCount:
        data = pickle.load(fCount)
    gram_count, unicount, bicount, tricount = data[0] ,data[1] ,data[2],data[3]
except:
    gram_count, unicount, bicount, tricount = kgram.initCount2()

def kneser_key(prepreviousword,previousword,word,nextword,nextnextword,vocab_corpus,theta):

    pre_other_pair = sum(list(map(lambda x:gram_count[x] if x in gram_count else 0,[previousword+" "+k for k in vocab_corpus])))
    pre_current_pair = gram_count[previousword+" "+word] if previousword+" "+word in gram_count else 0
    if pre_other_pair == 0: pre_other_pair = 1e-7
    if pre_current_pair == 0: pre_current_pair = 1e-7
    other_current_pair = sum(list(map(lambda x:gram_count[x] if x in gram_count else 0,[k+" "+word for k in vocab_corpus])))
    lamda = theta*pre_current_pair/(pre_other_pair)
    P_pre = max(pre_current_pair-theta,0)/(pre_other_pair) + lamda * other_current_pair/bicount

    other_next_pair = sum(
        list(map(lambda x: gram_count[x] if x in gram_count else 0, [k + " " + nextword for k in vocab_corpus])))
    current_next_pair = gram_count[word + " " + nextword] if word + " " + nextword in gram_count else 0
    if other_next_pair == 0: other_next_pair = 1e-7
    if current_next_pair == 0: current_next_pair = 1e-7
    current_other_pair = sum(
        list(map(lambda x: gram_count[x] if x in gram_count else 0, [word + " " + k for k in vocab_corpus])))
    lamda = theta * current_next_pair / (other_next_pair)
    P_next = max(current_next_pair - theta, 0) / (other_next_pair) + lamda * current_other_pair / bicount

    if prepreviousword == "":
        return np.log(P_pre) + np.log(P_next)

    preprepairs = [prepreviousword+" "+k for k in vocab_corpus] # pair of (prepre, vocab)
    prepre_other_pair =  sum(list(map(lambda x:gram_count[x] if x in gram_count else 0,[k+" "+g for k in preprepairs for g in vocab_corpus])))
    prepre_current_pair = sum(map(lambda x: gram_count[x+" "+word] if x+" "+word in gram_count else 0, preprepairs))
    if prepre_other_pair == 0: prepre_other_pair = 1e-7
    if prepre_current_pair == 0: prepre_current_pair = 1e-7
    precurrentpairs = [k+" "+word for k in vocab_corpus]
    preother_current_pair = sum(list(map(lambda x:gram_count[x] if x in gram_count else 0,[k+" "+g for k in vocab_corpus for g in precurrentpairs])))
    lamda = theta * prepre_current_pair / (prepre_other_pair)
    P_prepre = max(prepre_current_pair - theta, 0) / (prepre_other_pair) + lamda * preother_current_pair / bicount

    postpostpairs = [k + " " + postpostIndex for k in vocab_corpus]  # pair of (vocab,postpost)
    postpost_other_pair = sum(list(map(lambda x: gram_count[x] if x in gram_count else 0,
                                     [k + " " + g for k in vocab_corpus for g in postpostpairs])))
    postpost_current_pair = sum(
        map(lambda x: gram_count[word + " " + x] if x + " " + word in gram_count else 0, postpostpairs))
    if postpost_other_pair == 0: postpost_other_pair = 1e-7
    if postpost_current_pair == 0: postpost_current_pair = 1e-7
    postcurrentpairs = [word + " " + k for k in vocab_corpus]
    current_otherpost_pair = sum(list(map(lambda x: gram_count[x] if x in gram_count else 0,
                                         [k + " " + g for k in postcurrentpairs for g in vocab_corpus])))
    lamda = theta * postpost_current_pair / (postpost_other_pair)
    P_nextnext = max(postpost_current_pair - theta, 0) / (postpost_other_pair) + lamda * current_otherpost_pair / bicount
    return np.log(P_pre)+np.log(P_next)+np.log(P_prepre)+np.log(P_nextnext)



def add_k(previousword, word, nextword, vocabCount, lamda):
    previousword, word, nextword = previousword, word, nextword
    if previousword + " " + word in gram_count and previousword in gram_count:
        P_word_given_previous = (gram_count[previousword + " " + word] + lamda) / (gram_count[
            previousword] + lamda * vocabCount)
    elif previousword + " " + word in gram_count:
        P_word_given_previous = (gram_count[previousword + " " + word] + lamda) / (lamda * vocabCount)
    elif previousword in gram_count:
        P_word_given_previous = lamda / (gram_count[previousword] + lamda * vocabCount)
    else:
        P_word_given_previous = 1 / vocabCount
    if word + " " + nextword in gram_count and word in gram_count:
        P_next_given_word = (gram_count[word + " " + nextword] + lamda) / (gram_count[word] + lamda * vocabCount)
    elif word + " " + nextword in gram_count:
        P_next_given_word = (gram_count[word + " " + nextword] + lamda) / (lamda * vocabCount)
    elif word in gram_count:
        P_next_given_word = lamda / (gram_count[word] + lamda * vocabCount)
    else:
        P_next_given_word = 1 / vocabCount

    return np.log(P_word_given_previous) + np.log(P_next_given_word)

def P(word,lamda,vocabCount):
    if word in gram_count:
        return (gram_count[word] + lamda)/(lamda * vocabCount)
    else: return (1 / vocabCount)

#
def search_trie(trie, word):
    t = trie
    for c in word:
        if c not in t:
            return False
        else:
            t = t[c]
    if END not in t:
        return False
    return True

def get_candidate(trie, word, edit_distance=1):
    que = deque([(trie, word, "", edit_distance)])
    while que:
        trie, word, path, edit_distance = que.popleft()
        if word == "":
            if END in trie:
                yield path
            if edit_distance > 0:
                for k in trie:
                    if k != END:
                        # 词尾增加字母
                        que.appendleft((trie[k], "", path + k, edit_distance - 1))
        else:
            if edit_distance > 0:
                for k in trie.keys() - {word[0], END}:
                    # 词中增加字母
                    que.append((trie[k], word, path + k, edit_distance-1))
                    # 词中替换字母
                    que.append((trie[k], word[1:], path + k, edit_distance-1))
                # 词中删除字母
                que.append((trie, word[1:], path, edit_distance-1))
                # 词内交换字母
                if len(word) > 1:
                    que.append((trie, word[1] + word[0] + word[2:], path, edit_distance-1))
            if word[0] in trie:
                que.append((trie[word[0]],word[1:],path+word[0],edit_distance))


start = time.time()
# load the data
datapath = "./testdata.txt"
vocabpath = "./vocab.txt"
vocabfile = open(vocabpath, "r")
vocabs = []
vocab = vocabfile.readline()
while (vocab):
    vocabs.append(vocab[:-1])
    vocab = vocabfile.readline()
# make a trie
trie = {}
END = "$"

for word in vocabs:
    t = trie
    for c in word:
        if c not in t:
            t[c] = {}
        t = t[c]
    t[END] = {}

datafile = open(datapath, "r")
datalines = []
for i in range(1000):
    dataline = datafile.readline().split('\t')
    datalines.append(dataline)
datafile.close()

end = time.time()
print("time of loading data",end-start)
# begin correction
start = time.time()

n = 100
fres = open("result.txt", "w")
corpus_raw_text = reuters.sents(categories=reuters.categories())
corpus_text = []
for sents in corpus_raw_text:
    sents = ['<s>'] + sents + ['</s>']
    # remove string.punctuation
    for words in sents[::]:  # use [::] to remove the continuous ';' ';'
        if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):
            sents.remove(words)
    corpus_text.extend(sents)
vocab_corpus = {}.fromkeys(corpus_text).keys()
vocab_corpus = list(vocab_corpus)
vocabCount = len(vocab_corpus)
corpus_str = ' '.join(corpus_text)
end = time.time()
print("time in making trie",end-start)



sentenceList = [dataline[2][:-1] for dataline in datalines]
start = time.time()
for t in range(n):
    sentence = datalines[t][2]
    wordList = nltk.word_tokenize(sentenceList[t])
    for word in wordList:
        if (word in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):
            wordList.remove(word)
    wordList.insert(0, "<s>")
    wordList.append("</s>")
    miswords = set(filter(lambda word: word not in vocabs,
                          wordList[1:-1]))
    if (t+1)%100 == 0 :
        print("current", t + 1)
    for i in range(1, len(wordList) - 2):
        word = wordList[i]
    # print(miswords)

    for misword in miswords:
        miswordIndex = wordList.index(misword)
        preIndex = miswordIndex - 1
        postIndex = miswordIndex + 1
        prepreIndex = miswordIndex - 2
        postpostIndex = miswordIndex + 2
        resultset = set(get_candidate(trie,misword,1))
        resultset = list(resultset)

        if not resultset:
            resultset = list(set(get_candidate(trie,misword,2)))
        if len(resultset) == 1:
            result = resultset.pop()
            sentence = sentence.replace(misword, result)
            continue

        # for word in wordList[1:-1]:
        #     candidates = list(get_candidate(trie, word, 1))
        #     candidates.append(word)
        #     wordIndex = wordList.index(misword)
        #     wordpreIndex = wordIndex - 1
        #     wordpostIndex = wordIndex + 1
        #     prob = []
        #     for candidate in candidates:
        #         Plm = add_k(wordList[wordpreIndex], candidate, wordList[wordpostIndex], vocabCount, 0.01)
        #         Pres = Plm + np.log(Pxw(word, candidate, corpus_text, corpus_str, 0.01)) - np.log(P(candidate,0.01,vocabCount))
        #         prob.append(Pres)
        #     if word == candidates[prob.index(min(prob))]:
        #         print("findone",t,word)
        #         sentence.replace(word, candidates[prob.index(max(prob))])

        prob = []

        for Word in resultset:

            # Plm = add_k(wordList[preIndex], Word, wordList[postIndex], vocabCount, 0.01)
            Plm = kneser_key("", wordList[preIndex], Word, wordList[postIndex], "", vocab_corpus, 0.75)
            # if prepreIndex == -1 or postpostIndex == len(wordList):
            #     Plm = kneser_key("",wordList[preIndex],Word,wordList[postIndex],"",vocab_corpus,0.75)
            # else:
            #     Plm = kneser_key(wordList[prepreIndex],wordList[preIndex], Word, wordList[postIndex],wordList[postpostIndex], vocab_corpus, 0.75)
            Pres = Plm + np.log(Pxw(misword, Word, corpus_text,corpus_str,0.01))
            prob.append(Pres)
        # print(resultset)

        sentence = sentence.replace(misword, resultset[prob.index(max(prob))])


    result = str(t + 1) + "\t" + sentence
    fres.writelines(result)
end = time.time()
print("time in spellCorrecting", end - start)
fres.close()
eval.evaluate(n)

import nltk
from collections import Counter
import pickle
from nltk.corpus import reuters

def words(text):
    return Counter(nltk.word_tokenize(text.lower()))


def initCount():
    f = open("reutersCorpus.txt")
    wordsCount = words(f.read())
    wordPairCount = {}
    wordTriCount = {}
    f.seek(0)
    line = f.readline().lower()
    while line:
        wordlist = nltk.word_tokenize(line)
        wordlist.insert(0, "<s>")
        wordlist.append("</s>")
        for i in range(len(wordlist) - 1):
            if (wordlist[i], wordlist[i + 1]) in wordPairCount.keys():
                wordPairCount[(wordlist[i], wordlist[i + 1])] += 1
            else:
                wordPairCount[(wordlist[i], wordlist[i + 1])] = 1
            if (wordlist[i - 1], wordlist[i], wordlist[i + 1]) in wordTriCount.keys():
                wordTriCount[wordlist[i - 1], wordlist[i], wordlist[i + 1]] += 1
            else:
                wordTriCount[wordlist[i - 1], wordlist[i], wordlist[i + 1]] = 1
        line = f.readline().lower()
    with open("Count.pk", "wb") as fCount:
        pickle.dump([wordsCount, wordPairCount, wordTriCount], fCount)
    return wordsCount, wordPairCount, wordTriCount

def initCount2():
    corpus_raw_text = reuters.sents(categories=reuters.categories())
    gram_count = {}
    count  = [0,0,0]
    for sents in corpus_raw_text:
        sents = ['<s>'] + sents + ['</s>']
        # remove string.punctuation
        for words in sents[::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):
                sents.remove(words)

        # count the n-gram
        for n in range(1, 3):  # only compute 1/2/3-gram
            if (len(sents) <= n):  # 'This sentence is too short!'
                continue
            else:
                for i in range(n, len(sents) + 1):
                    gram = sents[i - n: i]  # ['richer', 'fuller', 'life']
                    key = ' '.join(gram)  # richer fuller life
                    count[n] = count[n] + 1
                    if (key in gram_count):  # use dict's hash
                        gram_count[key] += 1
                    else:
                        gram_count[key] = 1
    with open("Count.pk", "wb") as fCount:
        pickle.dump([gram_count], fCount)
    return gram_count , count[0], count[1], count[2]
# -*- coding: utf-8 -*-
import difflib
import numpy as np
import pickle

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

def initMatrix():
    f = open("spell-errors.txt","r")
    dataline = f.readline().lower()
    datalines = []
    while dataline:
        if "-" in dataline:
            dataline = dataline.replace("-","#") # replace "-" with "#" to use difflib easily
        datalines.append(dataline)
        dataline = f.readline().lower()
    f.close()

    Word2Error = {}
    for dataline in datalines:
        originWord = dataline.split(": ")[0]
        errorWords = dataline.split(": ")[1][:-1].split(", ")
        # print(errorWords)
        Word2Error[originWord] = errorWords
    #initialize those Matrix
    delMatrix = {}
    insMatrix = {}
    subMatrix = {}
    transMatrix = {}
    CountMatrix = {}
    CountMatrix2 = {}
    for x in 'abcdefghijklmnopqrstuvwxyz\'_#0123456789 ' :
        for y in 'abcdefghijklmnopqrstuvwxyz\'_#?0123456789 ':
            delMatrix[(x,y)] = 0
            insMatrix[(x, y)] = 0
            subMatrix[(x, y)] = 0
            transMatrix[(x, y)] = 0
            CountMatrix2[(x, y)] = 0
        CountMatrix[x] = 0
    d = difflib.Differ()
    # with the help of difflib, the insert and delete will be performed in "+" ,"-" in function result
    for originWord in Word2Error:
        for i in range(len(originWord)):
            CountMatrix[originWord[i]] += 1
            if i != len(originWord)-1:
                CountMatrix2[(originWord[i],originWord[i+1])] += 1
        for errorWord in Word2Error[originWord]:
            if edit_distance(originWord,errorWord)<=2:
                # print("1",originWord,"2",errorWord,list(d.compare(originWord,errorWord)))
                diff = "".join(list(d.compare(originWord,errorWord)))
                diff = "".join(diff.split(" "))
                # print(diff)
                if diff.count("+") == 1:
                    addIndex = diff.find("+")
                    if diff.count("-") == 1:
                        deleteIndex = diff.find("-")
                        if addIndex - deleteIndex == 2 or deleteIndex - addIndex == 2:
                            print(originWord,errorWord,"that is sub",(diff[addIndex+1],diff[deleteIndex+1]))
                            subMatrix[(diff[deleteIndex+1],diff[addIndex+1])] += 1
                        elif (addIndex - deleteIndex == 3 or deleteIndex - addIndex == 3)and diff[addIndex + 1] == diff[
                deleteIndex + 1]:
                            transMatrix[(diff[deleteIndex-1],diff[deleteIndex+1])] += 1
                            print(originWord, errorWord, "that is trans",(diff[deleteIndex+1],diff[deleteIndex-1]))
                        else: continue # none of those four situation
                    elif diff.count("-") == 0:
                        # print((diff[addIndex-1],diff[addIndex+1]))
                        print(originWord, errorWord, "that is ins",(diff[addIndex-1],diff[addIndex+1]))
                        insMatrix[(diff[addIndex-1],diff[addIndex+1])] += 1
                    else:
                        print(originWord,errorWord,"that is none")
                        continue # none of those four situation
                elif diff.count("-") == 1:
                    deleteIndex = diff.find("-")
                    delMatrix[(diff[deleteIndex-1],diff[deleteIndex+1])] += 1
                    print(originWord, errorWord, "that is del", (diff[deleteIndex-1],diff[deleteIndex+1]))
                else:
                    print(originWord, errorWord, "that is none")
                    continue # none of those four situation
    with open("Matrix.pk","wb") as Matrixf:
        pickle.dump([delMatrix,insMatrix,subMatrix,transMatrix,CountMatrix,CountMatrix2],Matrixf)
        # print("trans",transMatrix)
    return delMatrix,insMatrix,subMatrix,transMatrix,CountMatrix,CountMatrix2


def initMatrix2():
    f = open("count_1edit.txt", "r")
    line = f.readline()
    while(line):
        item = line.split("\t")
        if "\n" in item[1]:
            item[1] = item[1][:-1]

        letters = (item[0]).split("|")
        delMatrix = {}
        insMatrix = {}
        subMatrix = {}
        transMatrix = {}
        if len(letters[0])==1 and len(letters[1])==1:
                subMatrix[(letters[0]+"|"+ letters[1])] = int(item[1])
        if len(letters[0])==1 and len(letters[1])==2:
                delMatrix[(letters[0]+"|"+ letters[1])] = int(item[1])
        if len(letters[0])==2 and len(letters[1])==1:
                insMatrix[(letters[0]+"|"+ letters[1])] = int(item[1])
        if len(letters[0])==2 and len(letters[1])==2:
                transMatrix[(letters[0]+"|"+ letters[1])] = int(item[1])
        line = f.readline()
    with open("Matrix.pk","wb") as Matrixf:
        pickle.dump([delMatrix,insMatrix,subMatrix,transMatrix],Matrixf)
        # print("trans",transMatrix)
    return delMatrix,insMatrix,subMatrix,transMatrix
# initMatrix2()





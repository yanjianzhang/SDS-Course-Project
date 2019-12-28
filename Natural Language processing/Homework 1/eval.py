import nltk
def evaluate(n):
    anspath = './ans.txt'
    resultpath = './result.txt'
    ansfile = open(anspath, 'r')
    resultfile = open(resultpath, 'r')
    count = 0
    for i in range(n):
        ansline = ansfile.readline().split('\t')[1]
        ansset = set(nltk.word_tokenize(ansline))

        resultline = resultfile.readline().split('\t')[1]
        resultset = set(nltk.word_tokenize(resultline))

        if ansset == resultset:
            count += 1
    print("Accuracy is : %.3f%%" % (count * 1.000 / n))
if __name__ == '__main__':
    n = 1000
    anspath='./ans.txt'
    resultpath='./result.txt'
    ansfile=open(anspath,'r')
    resultfile=open(resultpath,'r')
    count=0
    for i in range(n):
        ansline=ansfile.readline().split('\t')[1]
        ansset=set(nltk.word_tokenize(ansline))
        resultline=resultfile.readline().split('\t')[1]
        resultset=set(nltk.word_tokenize(resultline))

        if ansset==resultset:
            count+=1
    print("Accuracy is : %.3f%%" % (count*1.000/n))

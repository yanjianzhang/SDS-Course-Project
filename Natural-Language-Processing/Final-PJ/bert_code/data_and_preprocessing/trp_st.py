# import json
import codecs

# with open("openIE_dev.json", encoding="utf-8") as f:
#     trp_dev = json.load(f)

####### dex
trp_dev = []
# dev_file = open('openIE_dev.json')
dev_file = open('dev.txt')
for line in dev_file:
    trp_dev.append(eval(line))

# print(trp_dev[0]['q_trp'][0])

dev_sent = []
for i in range(len(trp_dev)):
    q = trp_dev[i]['q_trp']
    question = ""
    if q:
        for j in range(len(q)):
            trp_q = q[j]
            if j == 0:
                question = trp_q['subject']+' '+trp_q['relation']+' '+trp_q['object']
            else:
                question += ' [SEP] '+trp_q['subject']+' '+trp_q['relation']+' '+trp_q['object']

    a = trp_dev[i]['a_trp']
    answer = ""
    if a:
        for j in range(len(a)):
            trp_a = a[j]
            if j == 0:
                answer = trp_a['subject']+' '+trp_a['relation']+' '+trp_a['object']
            else:
                answer += ' [SEP] '+trp_a['subject']+' '+trp_a['relation']+' '+trp_a['object']
    line = {'Q':question, 'A':answer}
    dev_sent.append(line)

# print(dev_sent)
# output file
f = codecs.open('dev_sent.tsv', 'w', 'utf8')
for i in range(len(dev_sent)):
    f.writelines(str(dev_sent[i])+'\n')
f.close()

####### train 

trp_train = []
# train_file = open('openIE_train.json')
train_file = open('train.txt')
for line in train_file:
    trp_train.append(eval(line))

# print(trp_train[0]['q_trp'][0])

train_sent = []
for i in range(len(trp_train)):
    q = trp_train[i]['q_trp']
    question = ""
    if q:
        for j in range(len(q)):
            trp_q = q[j]
            if j == 0:
                question = trp_q['subject']+' '+trp_q['relation']+' '+trp_q['object']
            else:
                question += ' [SEP] '+trp_q['subject']+' '+trp_q['relation']+' '+trp_q['object']

    a = trp_train[i]['a_trp']
    answer = ""
    if a:
        for j in range(len(a)):
            trp_a = a[j]
            if j == 0:
                answer = trp_a['subject']+' '+trp_a['relation']+' '+trp_a['object']
            else:
                answer += ' [SEP] '+trp_a['subject']+' '+trp_a['relation']+' '+trp_a['object']
    line = {'Q':question, 'A':answer}
    train_sent.append(line)

# print(train_sent)
# output file
f = codecs.open('train_sent.tsv', 'w', 'utf8')
for i in range(len(train_sent)):
    f.writelines(str(train_sent[i])+'\n')
f.close()

####### test

trp_test = []
test_file = open('openIE-test.json')
# test_file = open('test.txt')
for line in test_file:
    trp_test.append(eval(line))

# print(trp_test[0]['q_trp'][0])

test_sent = []
for i in range(len(trp_test)):
    q = trp_test[i]['q_trp']
    question = ""
    if q:
        for j in range(len(q)):
            trp_q = q[j]
            if j == 0:
                question = trp_q['subject']+' '+trp_q['relation']+' '+trp_q['object']
            else:
                question += ' [SEP] '+trp_q['subject']+' '+trp_q['relation']+' '+trp_q['object']

    a = trp_test[i]['a_trp']
    answer = ""
    if a:
        for j in range(len(a)):
            trp_a = a[j]
            if j == 0:
                answer = trp_a['subject']+' '+trp_a['relation']+' '+trp_a['object']
            else:
                answer += ' [SEP] '+trp_a['subject']+' '+trp_a['relation']+' '+trp_a['object']
    line = {'Q':question, 'A':answer}
    test_sent.append(line)

# print(test_sent)
# output file
f = codecs.open('test_sent.tsv', 'w', 'utf8')
for i in range(len(test_sent)):
    f.writelines(str(test_sent[i])+'\n')
f.close()
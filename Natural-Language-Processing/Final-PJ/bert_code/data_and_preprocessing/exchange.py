import json
import codecs

# with open("openIE_dev.json", encoding="utf-8") as f:
#     trp_dev = json.load(f)

####### dex
trp_dev = []
dev_file = open('dev_ex_trp.json')
# dev_file = open('dev.txt')
for line in dev_file:
    trp_dev.append(eval(line))

ori_dev = []
ori_dev_file = open('dev_reason.tsv')
# dev_file = open('dev.txt')
for line in ori_dev_file:
    foo = line.strip('\n')
    ori_dev.append(foo)

# print(trp_dev[0]['q_trp'][0])

for i in range(len(trp_dev)):
    question = ori_dev[i//3].split("\t")
    trp_dev[i]['question'] = question[2]

"""
# print(dev_sent)
# output file
f = codecs.open('train_ex_final.tsv', 'w', 'utf8')
for i in range(len(trp_dev)):
    f.writelines(str(trp_dev[i])+'\n')
f.close()
"""

with open('dev.reason_openIE.jsonl', "w") as fout:
    for i in range(len(trp_dev)):
        fout.write(json.dumps(trp_dev[i]) + "\n")

####### train 
"""
trp_train = []
train_file = open('train_trp.json')
# train_file = open('train.txt')
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
test_file = open('test_trp.json')
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
"""
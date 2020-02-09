import codecs

###### dev
dev2 = []
dev2_file = open('dev2.tsv')
for line in dev2_file:
    foo = line.strip('\n')
    dev2.append(foo)

dev_trp = []
dev_trp_file = open('dev_sent.tsv')
for line in dev_trp_file:
    dev_trp.append(eval(line))

final = []
for i in range(len(dev2)):
    dev2[i] = dev2[i].split("\t")
    index = dev2[i][0]
    ans = dev2[i][1]
    ques = dev_trp[3*i]['Q']
    sent = index+'\t'+ans+'\t'+ques
    for j in range(3):
        sent += '\t'+ dev_trp[3*i+j]['A']
    final.append(sent)

# print(final)
# output file
f = codecs.open('dev_trp_std.tsv', 'w', 'utf8')
for i in range(len(final)):
    f.writelines(str(final[i])+'\n')
f.close()

###### train
train2 = []
train2_file = open('train2.tsv')
for line in train2_file:
    foo = line.strip('\n')
    train2.append(foo)

train_trp = []
train_trp_file = open('train_sent.tsv')
for line in train_trp_file:
    train_trp.append(eval(line))

final = []
for i in range(len(train2)):
    train2[i] = train2[i].split("\t")
    index = train2[i][0]
    ans = train2[i][1]
    ques = train_trp[3*i]['Q']
    sent = index+'\t'+ans+'\t'+ques
    for j in range(3):
        sent += '\t'+ train_trp[3*i+j]['A']
    final.append(sent)

# print(final)
# output file
f = codecs.open('train_trp_std.tsv', 'w', 'utf8')
for i in range(len(final)):
    f.writelines(str(final[i])+'\n')
f.close()

###### test
test2 = []
test2_file = open('test2.tsv')
for line in test2_file:
    foo = line.strip('\n')
    test2.append(foo)

test_trp = []
test_trp_file = open('test_sent.tsv')
for line in test_trp_file:
    test_trp.append(eval(line))

final = []
for i in range(len(test2)):
    test2[i] = test2[i].split("\t")
    index = test2[i][0]
    ans = test2[i][1]
    ques = test_trp[3*i]['Q']
    sent = index+'\t'+ans+'\t'+ques
    for j in range(3):
        sent += '\t'+ test_trp[3*i+j]['A']
    final.append(sent)

# print(final)
# output file
f = codecs.open('test_trp_std.tsv', 'w', 'utf8')
for i in range(len(final)):
    f.writelines(str(final[i])+'\n')
f.close()
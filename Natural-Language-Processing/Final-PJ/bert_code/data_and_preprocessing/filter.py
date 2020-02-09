from nltk.metrics import edit_distance
import codecs

######## dev

# input file 

dev2 = []
dev2_file = open('dev2.tsv')
for line in dev2_file:
    foo = line.strip('\n')
    dev2.append(foo)

answer = []
for i in range(len(dev2)):
    dev2[i] = dev2[i].split("\t")
    ans = dev2[i][1]
    if ans == 'A':
        answer.append(dev2[i][3])
    elif ans == 'B':
        answer.append(dev2[i][4])
    elif ans == 'C':
        answer.append(dev2[i][5])

print(len(answer))

dev3 = []
dev3_file = open('dev3.tsv')
for line in dev3_file:
    foo = line.strip('\n')
    dev3.append(foo)

for i in range(len(dev3)):
    dev3[i] = dev3[i].split("\t")
    dev3[i] = dev3[i][2:5]

print(len(dev3))

# Choose the one with max edit_distance

reason = []
for i in range(len(answer)):
    max_dist = -1
    reason_index = 99
    for j in range(3):
        dist = edit_distance(answer[i],dev3[i][j])
        if dist > max_dist:
            max_dist = dist
            reason_index = j
    reason.append(dev3[i][reason_index])

# output file
f = codecs.open('dev_extra.tsv', 'w', 'utf8')
for i in range(len(reason)):
    f.writelines(dev2[i][0] + '\t' + dev2[i][1] + '\t' + dev2[i][2] + '\t' + dev2[i][3] + '\t' + dev2[i][4] + '\t' + dev2[i][5] + '\t' + reason[i] + '\n')
f.close()

# f = codecs.open('dev_reason.tsv', 'w', 'utf8')
# for i in range(len(reason)):
#     f.writelines(dev2[i][0] + '\t' + dev2[i][1] + '\t' + dev2[i][2] + ' [SEP] ' + reason[i] + '\t' + dev2[i][3] + '\t' + dev2[i][4] + '\t' + dev2[i][5] + '\n')
# f.close()


######## train

# input file 

train2 = []
train2_file = open('train2.tsv')
for line in train2_file:
    foo = line.strip('\n')
    train2.append(foo)

answer = []
for i in range(len(train2)):
    train2[i] = train2[i].split("\t")
    ans = train2[i][1]
    if ans == 'A':
        answer.append(train2[i][3])
    elif ans == 'B':
        answer.append(train2[i][4])
    elif ans == 'C':
        answer.append(train2[i][5])

print(len(answer))

train3 = []
train3_file = open('train3.tsv')
for line in train3_file:
    foo = line.strip('\n')
    train3.append(foo)

for i in range(len(train3)):
    train3[i] = train3[i].split("\t")
    train3[i] = train3[i][2:5]

print(len(train3))

# Choose the one with max edit_distance

reason = []
for i in range(len(answer)):
    max_dist = -1
    reason_index = 99
    for j in range(3):
        dist = edit_distance(answer[i],train3[i][j])
        if dist > max_dist:
            max_dist = dist
            reason_index = j
    reason.append(train3[i][reason_index])

# output file
f = codecs.open('train_extra.tsv', 'w', 'utf8')
for i in range(len(reason)):
    f.writelines(train2[i][0] + '\t' + train2[i][1] + '\t' + train2[i][2] + '\t' + train2[i][3] + '\t' + train2[i][4] + '\t' + train2[i][5] + '\t' + reason[i] + '\n')
f.close()

# f = codecs.open('train_reason.tsv', 'w', 'utf8')
# for i in range(len(reason)):
#     f.writelines(train2[i][0] + '\t' + train2[i][1] + '\t' + train2[i][2] + ' [SEP] ' + reason[i] + '\t' + train2[i][3] + '\t' + train2[i][4] + '\t' + train2[i][5] + '\n')
# f.close()

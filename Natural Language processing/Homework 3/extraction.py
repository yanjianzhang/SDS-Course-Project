from collections import defaultdict
import numpy as np
import codecs
class HMM():
	def __init__(self, type, mylambda):
		self.type = type
		self.mylambda = mylambda
		file = codecs.open(self.type + '_train.txt', 'r', 'utf8')
		self.start = defaultdict(int) # start label -> int
		self.labels = []
		self.transition = defaultdict(lambda: defaultdict(int)) # label -> label -> count
		self.emission = defaultdict(lambda: defaultdict(int)) # label -> word -> count

		preLabel = None
		for line in file.readlines():

			if line.strip():
				word, label = line.strip().split()
				if label not in self.labels:
					self.labels.append(label)

				if not preLabel:
					self.start[label] += 1
					self.emission[label][word] += 1
					preLabel = label
				else:
					self.transition[preLabel][label] += 1
					self.emission[label][word] += 1
					preLabel = label
			else:
				preLabel = None

		self.trans_prob = defaultdict(lambda: defaultdict(float))
		self.emit_prob = defaultdict(lambda: defaultdict(float))
		self.start_prob = {}

		# Smooth with add-lambda
		start_sum = sum(self.start.values())
		start_class = len(self.start.keys())
		for label in self.labels:
			self.start_prob[label] = (self.start[label] if label in self.start else 0 + self.mylambda) /(start_sum + self.mylambda * start_class)
		self.start_prob['Unknown label'] = self.mylambda / (
						start_sum + self.mylambda * start_class)


		for label in self.transition.keys():
			label_sum = sum(self.transition[label].values())
			label_class = len(self.emission[label].keys())
			for next_label in self.transition[label].keys():
				self.trans_prob[label][next_label] = (self.transition[label][next_label] + self.mylambda) / (label_sum + self.mylambda * label_class)
				self.trans_prob[label]['Unknown label'] = self.mylambda / (
							label_sum + self.mylambda * label_class)

		for label in self.emission.keys():
			label_sum = sum(self.emission[label].values())
			label_class = len(self.emission[label].keys())
			for word in self.emission[label].keys():
				self.emit_prob[label][word] = (self.emission[label][word] + self.mylambda) / (label_sum + self.mylambda * label_class)
			self.emit_prob[label]['Unknown word'] = self.mylambda / (
						label_sum + self.mylambda * label_class)

	def predict(self, seq):
		time_state_prob = defaultdict(lambda: defaultdict(float))
		path_state = {}
		for label in self.labels:
			word = seq[0] if seq[0] in self.emit_prob[label] else 'Unknown word'
			time_state_prob[0][label] = np.log(self.start_prob[label]) +  np.log(self.emit_prob[label][word])
			path_state[label] = [label]
		for t in range(1, len(seq)):
			new_path =  {} # up to current t, the best path correspond to the label
			for label in self.labels:
				prob = {}
				for pre_label in  self.labels:
					word = seq[t] if seq[t] in self.emit_prob[label] else 'Unknown word'
					prob_condition = np.log(self.trans_prob[pre_label][label if label in self.trans_prob[pre_label] else 'Unknown label']) + np.log(self.emit_prob[label][word])
					prob_prior = time_state_prob[t-1][pre_label]
					prob[pre_label] = prob_prior + prob_condition
				pre_label_max, prob_max = max(prob.items(), key= lambda x: x[1])
				time_state_prob[t][label] = prob_max
				new_path[label] = path_state[pre_label_max] + [label]
			path_state = new_path
		label_max, prob_max = max([(label, time_state_prob[len(seq)-1][label]) for label in self.labels], key=lambda x:x[1])

		return path_state[label_max]


	def test(self):
		test_file = codecs.open(self.type + '_test.txt', 'r', 'utf8')

		result = []
		seq = []
		lines = test_file.readlines()
		showseq = []
		showpath = []
		for index, line in enumerate(lines):

			if line.strip():  # is not a blank line
				words = line.strip().split()
				seq.append(words[0])
			else:
				path = self.predict(seq)
				result.extend(path)
				showpath = path
				showseq = seq
				seq = []

			if index%1000 == 0:
				print(str(index)+"/"+str(len(lines)))
				print("seq", showseq)
				print("path",showpath)


		file = codecs.open(self.type + '_result.txt', 'w', 'utf8')
		iter = 0

		for line in lines:
			if line.strip():  # is not a blank line
				file.writelines(line.strip() + '\t' + result[iter] + '\n')
				iter += 1
			else:
				file.writelines('\n')
				continue
		test_file.close()
		file.close()

	def evaluation(self):
		f = codecs.open(self.type + '_result.txt', 'r', 'utf8')
		result = f.readlines()
		f.close()

		TP, FP, TN, FN, type_correct, sum = 0, 0, 0, 0, 0, 0
		print(TP, FP, TN, FN, type_correct, sum)

		for word in result:
			if word.strip():
				sum += 1
				li = word.strip().split()
				if li[1] != 'O' and li[2] != 'O':
					TP += 1
					if li[1] == li[2]:
						type_correct += 1
				if li[1] != 'O' and li[2] == 'O':
					FN += 1
				if li[1] == 'O' and li[2] != 'O':
					FP += 1
				if li[1] == 'O' and li[2] == 'O':
					TN += 1

		recall = TP / (TP + FN)
		precision = TP / (TP + FP)
		accuracy = (TP + TN) / sum
		F1 = 2 * precision * recall / (precision + recall)

		print('=====' + self.type + ' labeling result=====')
		print("type_correct: ", round(type_correct / TP, 4))
		print("accuracy: ", round(accuracy, 4))
		print("precision: ", round(precision, 4))
		print("recall: ", round(recall, 4))
		print("F1: ", round(F1, 4))


if __name__ == '__main__':
	Hmm = HMM('argument', 0.01)  # manually set: 'argument' or 'trigger'
	Hmm.test()
	Hmm.evaluation()

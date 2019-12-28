# import numpy as np
# from numpy import exp
# from numpy.random import normal
# from numpy import array
# # from numpy import dot
# from numpy import zeros
# from numpy import load
# from logUtil import *
from weight import hidden_weight
from weight import output_weight
from listTool import *





def sigmoid_derivative(value):
	if value.any() >= 0:
		s = 1.0 / (1 + exp(-value))
	else:
		s = exp(value) / (1 + exp(value))
	ds = s * (1 - s)
	return ds


class NeuralNetwork(object):
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		try:
			self.weights_input_to_hidden = hidden_weight
			# print('len1:', len(self.weights_input_to_hidden))
			# print('len2', len(self.weights_input_to_hidden[1]))
		except:
			print("hidden weight not found")
		try:
			self.weights_hidden_to_output = output_weight
		except:
			print("output weight not found")
		# self.weights_input_to_hidden = zeros((self.hidden_nodes, self.input_nodes))
		#
		# self.weights_hidden_to_output = zeros((self.output_nodes, self.hidden_nodes))

		self.lr = learning_rate

		# Activation function is the sigmoid function
		# self.activation_function = (lambda x: 1/(1 - np.exp(-x)))
		self.activation_function = lambda x: list_fn(x, sigmoid)

	def train(self, inputs_list1, inputs_list2, reward, gamma):

		# Convert inputs list to 2d array
		# print(inputs_list1, inputs_list2)
		inputs1 = inputs_list1  # V_t shape [feature_diemension, 1]
		inputs2 = inputs_list2  # V_t+1 shape [feature_diemension, 1]
		# targets = np.array(targets_list, ndmin=2).T
		# Forward pass
		# TODO: Hidden layer
		# print(self.weights_input_to_hidden, inputs1)
		hidden_inputs1 = mydot(self.weights_input_to_hidden, inputs1)  # signals into hidden layer
		# print(hidden_inputs1)
		hidden_outputs1 = self.activation_function(hidden_inputs1)  # signals from hidden layer
		hidden_inputs2 = mydot(self.weights_input_to_hidden, inputs2)  # signals into hidden layer
		hidden_outputs2 = self.activation_function(hidden_inputs2)  # signals from hidden layer

		# activation y = x
		final_inputs1 = mydot(self.weights_hidden_to_output, hidden_outputs1)  # signals into final output layer

		final_outputs1 = self.activation_function(final_inputs1)  # signals from final output layer
		final_inputs2 = mydot(self.weights_hidden_to_output, hidden_outputs2)  # signals into final output layer
		final_outputs2 = self.activation_function(final_inputs2)  # signals from final output layer
		### Backward pass ###

		# Output layer error is the difference between desired target and actual output.
		# output_errors = (final_outputs1-reward - gamma*final_outputs2)
		# (reward + gamma * final_outputs2 - final_outputs1) * (final_outputs1 * (1 - final_outputs1)).T shape (1, 1)
		output_errors = mydot((add(reward, mydot(float(gamma), minus(final_outputs2,final_outputs1)))) , (mydot(final_outputs1, minus(1, final_outputs1))))

		# errors propagated to the hidden layer shape (1 * 60)
		# hidden_errors = dot(output_errors, self.weights_hidden_to_output) * (hidden_outputs1 * (1 - hidden_outputs1)).T
		hidden_errors = mydot(mydot(output_errors, self.weights_hidden_to_output) , mydot(hidden_outputs1 , minus(1 , hidden_outputs1)))
		# hidden_errors2 = (-gamma)*np.dot(output_errors, self.weights_hidden_to_output)*(hidden_outputs2*(1-hidden_outputs2)).T
		# logging.debug("\n1"+str(output_errors)+"\n2"+str(hidden_errors)+"\n3 pre"+str(self.weights_hidden_to_output))
		# Update the weights
		# update hidden-to-output weights with gradient descent step shape (1, 60)
		self.weights_hidden_to_output =  add(self.weights_hidden_to_output, mydot(mydot(output_errors, hidden_outputs1), self.lr))

		# update input-to-hidden weights with gradient descent step shape(60, 218)
		self.weights_input_to_hidden = add(self.weights_input_to_hidden, mydot(inputs1, mydot(hidden_errors , self.lr)))
		#logging.debug("\n4 post" + str(self.weights_hidden_to_output) + "\n5 post" + str(self.weights_input_to_hidden))

	#
	def run(self, inputs_list):
		# Run a forward pass through the network
		inputs = inputs_list
		# print('len3:',len(inputs))
		#### Implement the forward pass here ####
		# Hidden layer
		hidden_inputs = mydot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
		# print('NN', hidden_inputs)
		hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

		# Output layer
		# print('NN', hidden_outputs)
		final_inputs = mydot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
		# final_outputs = final_inputs # signals from final output layer
		# print('NN', final_inputs)
		final_outputs = self.activation_function(final_inputs)[0]  # signals from final output layer

		return final_outputs

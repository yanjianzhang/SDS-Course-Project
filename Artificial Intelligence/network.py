# import numpy as np
from numpy import exp
from numpy.random import normal
from numpy import array
from numpy import dot
# from numpy import zeros
from numpy import load
from logUtil import *
nameofrandom = 40

def sigmoid(value):
	if value.any() >= 0:
		return 1.0 / (1 + exp(-value))
	else:
		return exp(value) / (1 + exp(value))


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
			self.weights_input_to_hidden = load("D:/input2hidden+"+str(nameofrandom)+".model.npy")
		except:
			self.weights_input_to_hidden = normal(0.0, self.hidden_nodes ** -0.5,
			                                      (self.hidden_nodes, self.input_nodes))
		try:
			self.weights_hidden_to_output = load(
				"D:/hidden2output"+str(nameofrandom)+".model.npy")
		except:
			self.weights_hidden_to_output = normal(0.0, self.output_nodes ** -0.5,
			                                       (self.output_nodes, self.hidden_nodes))
		# self.weights_input_to_hidden = zeros((self.hidden_nodes, self.input_nodes))
		#
		# self.weights_hidden_to_output = zeros((self.output_nodes, self.hidden_nodes))

		self.lr = learning_rate

		# Activation function is the sigmoid function
		# self.activation_function = (lambda x: 1/(1 - np.exp(-x)))
		self.activation_function = (lambda x: sigmoid(x))

	def train(self, inputs_list1, inputs_list2, reward, gamma):

		# Convert inputs list to 2d array
		inputs1 = array(inputs_list1, ndmin=2).T  # V_t shape [feature_diemension, 1]
		inputs2 = array(inputs_list2, ndmin=2).T  # V_t+1 shape [feature_diemension, 1]
		# targets = np.array(targets_list, ndmin=2).T
		# Forward pass
		# TODO: Hidden layer
		hidden_inputs1 = dot(self.weights_input_to_hidden, inputs1)  # signals into hidden layer
		hidden_outputs1 = self.activation_function(hidden_inputs1)  # signals from hidden layer
		hidden_inputs2 = dot(self.weights_input_to_hidden, inputs2)  # signals into hidden layer
		hidden_outputs2 = self.activation_function(hidden_inputs2)  # signals from hidden layer

		# activation y = x
		final_inputs1 = dot(self.weights_hidden_to_output, hidden_outputs1)  # signals into final output layer

		final_outputs1 = self.activation_function(final_inputs1)  # signals from final output layer
		final_inputs2 = dot(self.weights_hidden_to_output, hidden_outputs2)  # signals into final output layer
		final_outputs2 = self.activation_function(final_inputs2)  # signals from final output layer
		### Backward pass ###

		# Output layer error is the difference between desired target and actual output.
		# output_errors = (final_outputs1-reward - gamma*final_outputs2)

		output_errors = (reward + gamma * final_outputs2 - final_outputs1) * (final_outputs1 * (1 - final_outputs1)).T
		# print(output_errors.shape,"out") shape (1, 1)
		# errors propagated to the hidden layer
		hidden_errors = dot(output_errors, self.weights_hidden_to_output) * (hidden_outputs1 * (1 - hidden_outputs1)).T
		# print(hidden_errors.shape)  shape (1 * 60)
		# hidden_errors2 = (-gamma)*np.dot(output_errors, self.weights_hidden_to_output)*(hidden_outputs2*(1-hidden_outputs2)).T
		# logging.debug("\n1"+str(output_errors)+"\n2"+str(hidden_errors)+"\n3 pre"+str(self.weights_hidden_to_output))
		# Update the weights
		# update hidden-to-output weights with gradient descent step
		self.weights_hidden_to_output += output_errors * hidden_outputs1.T * self.lr
		# print(self.weights_hidden_to_output.shape, "out2") shape (1, 60)
		# update input-to-hidden weights with gradient descent step
		self.weights_input_to_hidden += (inputs1 * hidden_errors * self.lr).T
		# print(self.weights_input_to_hidden.shape, "hidden") shape(60, 218)
		logging.debug("\n4 post" + str(self.weights_hidden_to_output) + "\n5 post" + str(self.weights_input_to_hidden))

	#
	def run(self, inputs_list):
		# Run a forward pass through the network
		inputs = array(inputs_list, ndmin=2).T

		#### Implement the forward pass here ####
		# Hidden layer
		hidden_inputs = dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

		# Output layer
		final_inputs = dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
		# final_outputs = final_inputs # signals from final output layer
		final_outputs = self.activation_function(final_inputs)  # signals from final output layer

		return final_outputs

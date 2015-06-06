import numpy as np
import math, operator, csv, sys, random

class neuralnetwork:

    # initialization parameter 'layer_sizes' must be a list of ints
	# reresenting the number of nodes in each layer e.g. a network
	# with 10 inputs, 5 nodes in the hidden layer and 3 outputs should have
	# layer_sizes = [10, 5, 3]
	def __init__(self, layer_sizes):
		self.num_inputs = layer_sizes[0]
		self.num_outputs = layer_sizes[-1] 
		self.layers = [[]] + [np.random.random_sample((i, j)) * 0.05 - 0.05 
					   for i, j in zip(layer_sizes[1:], layer_sizes[:-1])]

    # helper function for computing the activation function used in the
	# network. the default activation function is the sigmoid
	def __ff_helper(self, inputs, w_vec,
					act_func = (lambda o : (1.0 / (1.0 + math.exp(-o)))), 
					bias = 0.0):
		return act_func(np.asscalar(np.dot(inputs, w_vec)) - bias)					   

	# given an input, the function feeds forward this input using the
	# activation function in __ff__helper and returns the resulting outputs
	# in the network 
	def feed_forward(self, inputs):
		self.outputs_network = [inputs]

		for i in xrange(1, len(self.layers)):

			cur_out = []

			for node in self.layers[i]:
				# cur_out.append(self.__ff_helper(self.outputs_network[i - 1], node))
				temp = np.asscalr(np.dot(self.outputs_network[i - 1], node))
				cur_out.append((1.0) / (1.0 + math.exp(temp * -1)))
			self.outputs_network.append(cur_out)

		return self.outputs_network


	# given the network output resulting from ann.feed_forward (above)
	# and the index of the node corresponding to the correct output label
	# back_prop calculates the error for each node in the network,
	# given the network output.
	def back_prop(self, network, given_label):
		output = network[-1]
		labels = np.zeros((len(output),), dtype = np.float)
		labels[given_label] = 1.0
		output_errors = []

		for i in xrange(len(output)): 
			output_errors.append(output[i] * (1 - output[i]) * 
										(labels[i] - output[i]))

		errors = [output_errors]
		
		for i in reversed(xrange(1, len(network) - 1)):
			err_hidden = []
			for j in xrange(len(network[i])):
				cur_layer = network[i]
				err_index = len(network) - 2 - i
				err_temp = 0.0

				for node, error in zip(self.layers[i], errors[err_index]): 
					err_temp = err_temp + node[j] * error

				err_hidden.append(cur_layer[j] * (1 - cur_layer[j]) * err_temp)

			errors.append(err_hidden)				  

		return ([[]] + errors[::-1])

	# given the error for each node from ann.back_prop (above), this
	# function updates the weights for each node in the network
	def update_weights(self, learning_rate, outputs, errors):
		for i in xrange(1,len(outputs)):
			for j in xrange(len(outputs[i])):
				for n in xrange(len(outputs[i - 1])):
					temp = learning_rate * np.array(outputs[i - 1])
				delta_w = learning_rate * errors[i][j] * np.array(outputs[i - 1])
				self.layers[i][j] += delta_w
		return self.layers

	# given an ndarray of examples and a zero-dimensional array of labels
	# having the same length as the first dimension of the examples array,
	# trains a multilayer, feed-forward neural network and returns the
	# accuracy after the final epoch of training
	def train_multilayer_NN(self, examples, labels, 
							epochs = 1000, threshold = 1.0):
		for epoch in xrange(epochs):
			print 'epoch/iteration:', epoch
			num_correct = 0.0
			num_exs = examples.shape[1]
			dataset = zip(examples, labels)
			random.shuffle(dataset)
			for ex, label in dataset:
				ex = ex / 255.0
				# pass the input forward and calculate output at every neuron
				network_outputs = self.feed_forward(ex)
				index = np.argmax(network_outputs[-1]) 
			 	if index == label:
					num_correct += 1
				else:
					errors = self.back_prop(network_outputs, label)	
					self.update_weights(0.05, network_outputs, errors)	
			accuracy = num_correct/float(len(examples))
			print 'percent correct\t:\t', accuracy
			if (accuracy > threshold):
				break
		return accuracy
	
	# writes the network weights to a file
	def write_to_file(self, out_fp):
		writr = csv.writer(out_fp)
		writr.writerow([self.num_inputs] + [len(layer) for layer in self.layers[1:]])
		for layer in self.layers[1:]:
			for node in layer:
				writr.writerow(node)
			out_fp.write('\n')


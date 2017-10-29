import tensorflow as tf
import numpy as np
import optparse
import os
import shutil
import time
import random
import sys
import pickle
import glob
import matplotlib.pyplot as plt

from layers import *

from scipy.misc import imsave
from scipy.misc import imresize
from PIL import Image
from tqdm import tqdm
from numpy import genfromtxt


class regression():

	def run_parser(self):

		self.parser = optparse.OptionParser()

		self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
		self.parser.add_option('--batch_size', type='int', default=1, dest='batch_size')
		self.parser.add_option('--max_epoch', type='int', default=20, dest='max_epoch')
		self.parser.add_option('--feature_size', type='int', default=20, dest='feature_size')
		self.parser.add_option('--test', action="store_true", default=False, dest="test")
		self.parser.add_option('--model', type='string', default="reg", dest='model_type')
		self.parser.add_option('--dataset_dir', type='string', default="./datasets/apple", dest='dataset_dir')

	def initialize(self):

		self.run_parser()

		opt = self.parser.parse_args()[0]

		self.max_epoch = opt.max_epoch
		self.batch_size = opt.batch_size
		self.model = "cnn"
		self.to_test = opt.test
		self.load_checkpoint = False
		self.do_setup = True
		self.dataset_dir = opt.dataset_dir
		self.feature_size = opt.feature_size
		self.feature_depth = 1

		self.tensorboard_dir = "./output/" + self.model + "/tensorboard"
		self.check_dir = "./output/"+ self.model  +"/checkpoints"

	
	def load_dataset(self):

		my_data = genfromtxt('./datasets/AAPL.csv', delimiter=',')
		my_data = my_data[1:,1:]
		self.adj_close = my_data[:,4].reshape((1,-1))

		self.adj_close = self.adj_close

	def divide_data(self):

		size = np.shape(self.adj_close)

		self.num_train_days = (int)(0.9*size[1])

		# Dividing the data into test and train data
		# Here the train vector is 90% or original data
		# and rest in test data

		self.train_data = self.adj_close[:, :self.num_train_days]
		self.test_data = self.adj_close[:, self.num_train_days:]

		# Shape of self.train_data = [1, self.num_train_days]

		self.num_test_days = self.test_data.size

		# print(self.train_data.shape)
		
		# Changing the shape of training data for "cnn" model
		# from [1, self.num_train_days] -> [1(self.batch_size), self.num_train_days, 1(self.feature_depth)]
		
		if(self.model == "cnn"):
			self.train_data = np.reshape(self.train_data, [1, -1, 1])
			self.test_data = np.reshape(self.test_data, [1, -1, 1])

	

	def normalize_data(self, data):

		return (data - np.mean(data, axis=1, keepdims=True))/np.std(data, axis=1, keepdims=True)

	def get_inputs(self, itr, feature_size, mode="train"):

		if(self.model == "simple"):

			if(mode == "train"):
				mean_val, std_val = np.mean(self.train_data[:, itr:itr+feature_size], keepdims=True)
				return (self.train_data[:, itr:itr+feature_size] - mean_val)/std_val, (self.train_data[:, itr+feature_size:itr+feature_size+1] - mean_val)/std_val
			elif(mode == "test"):
				mean_val, std_val = np.mean(self.test_data[:, itr:itr+feature_size], keepdims=True)
				return (self.test_data[:, itr:itr+feature_size] - mean_val)/std_val, (self.test_data[:, itr+feature_size:itr+feature_size+1] - mean_val)/std_val

		if(self.model == "cnn"):

			if(mode == "train"):

				temp_data = self.normalize_data(self.train_data[:, itr:itr+feature_size+1])
				# print(temp_data.shape)
				return temp_data[:, :feature_size], np.reshape(temp_data[:, feature_size:feature_size+1],[self.batch_size,-1])
			
			elif(mode == "test"):

				temp_data = self.normalize_data(self.test_data[:, itr:itr+feature_size+1])
				return temp_data[:, itr:itr+feature_size], np.reshape(temp_data[:, itr+feature_size:itr+feature_size+1],[self.batch_size,-1])


	def simple_model_setup(self):

		with tf.variable_scope("model_simple") as scope:

			self.input_feature = tf.placeholder(tf.float32, [self.batch_size, self.feature_size])
			self.output_value = tf.placeholder(tf.float32, [self.batch_size, 1])

			o_l1 = tf.nn.relu(linear1d(self.input_feature, self.feature_size, 500, "layer1"))
			o_l2 = tf.nn.relu(linear1d(o_l1, 500, 250, "layer2"))
			self.pred = linear1d(o_l2, 250, 1, "layer3")


	def cnn_model_setup(self):

		with tf.variable_scope("model_cnn") as scope:

			self.input_feature = tf.placeholder(tf.float32, [None, self.feature_size, self.feature_depth])
			self.output_value = tf.placeholder(tf.float32, [None, 1])

			o_c1 = tf.nn.relu(general_conv1d(self.input_feature, 64, 2, 1, padding="same", name="conv1"))
			o_c1 = tf.layers.max_pooling1d(o_c1, 2, 2, padding="valid")
			# print(o_c1.get_shape().as_list())
			o_c2 = tf.nn.relu(general_conv1d(o_c1, 32, 2, 1, padding="same", name="conv2"))
			o_c2 = tf.layers.max_pooling1d(o_c2, 2, 2, padding="valid")
			# print(o_c2.get_shape().as_list())
			o_c2 = tf.reshape(o_c2,[self.batch_size, -1])
			# print(o_c2.get_shape().as_list())

			o_l1 = tf.nn.relu(linear1d(o_c2, 160, 250, "layer1"))
			# print(o_l1.get_shape().as_list())
			self.pred = linear1d(o_l1, 250, 1, "layer2")


	def model_setup(self):

		# Initially do_setup is set to true

		if(self.do_setup):

			with tf.variable_scope("model") as scope:

				if(self.model == "simple"):
					self.simple_model_setup()
				elif(self.model == "cnn"):
					self.cnn_model_setup()


		# Printing the trainable variables for the model

		self.model_vars = tf.trainable_variables()
		for var in self.model_vars: print(var.name, var.get_shape())

		# Once the model is done setting up. we set it to false 
		# so that we don't set it up again in future

		self.do_setup = False


	def loss_setup(self):

		self.loss = tf.reduce_mean(tf.squared_difference(self.pred, self.output_value))
		optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
		self.loss_optimizer = optimizer.minimize(self.loss)


	def train(self):

		self.model_setup()
		self.loss_setup()
		self.load_dataset()
		self.divide_data()

		print(self.num_train_days)


		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		
		if not os.path.exists(self.check_dir):
			os.makedirs(self.check_dir)
		

		with tf.Session() as sess:

			sess.run(init)
			writer = tf.summary.FileWriter(self.tensorboard_dir)
			writer.add_graph(sess.graph)

			if self.load_checkpoint:
				chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
				saver.restore(sess,chkpt_fname)

			for epoch in range(0, self.max_epoch):

				for itr in range(0, int(self.num_train_days - self.feature_size-1)):
					
					temp_input_feature, temp_output = self.get_inputs(itr, self.feature_size, "train")

					# print(temp_input_feature.shape, temp_output.shape)
					# sys.exit()

					_, temp_loss = sess.run([self.loss_optimizer, self.loss], 
						feed_dict={self.input_feature:temp_input_feature, self.output_value:temp_output})

					if(itr%100 == 0):
						print("In the epoch " + str(epoch) + " and the iteration " + str(itr) + " with a loss of " + str(temp_loss))

			sys.exit()

				# saver.save(sess,os.path.join(self.check_dir,"Regress"),global_step=epoch)


	def test(self):

		self.model_setup()
		self.loss_setup()
		self.load_dataset()
		self.divide_data()

		saver = tf.train.Saver()
		
		if not os.path.exists(self.check_dir):
			print("No checkpoint directory exist")
			sys.exit()

		model_output_list = []
		orig_output_list = []

		with tf.Session() as sess:

			chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
			saver.restore(sess, chkpt_fname)

			for itr in range(0, int(self.num_test_days - self.feature_size)):

				temp_input_feature, temp_output = self.get_inputs(itr, self.feature_size, "test")

				temp_loss, model_output = sess.run([self.loss, self.pred], 
						feed_dict={self.input_feature:temp_input_feature, self.output_value:temp_output})

				model_output_list.append(model_output[0][0])
				orig_output_list.append(temp_output[0][0])

		plt.plot(model_output_list)
		plt.plot(orig_output_list)
		plt.show()


def main():

	model = regression()
	model.initialize()

	
	if(model.to_test):
		model.test()
	else:
		model.train()


if __name__ == "__main__":
	main()


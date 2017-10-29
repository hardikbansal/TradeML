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
from utils import *

from scipy.misc import imsave
from scipy.misc import imresize
from PIL import Image
from tqdm import tqdm
from numpy import genfromtxt


class regression():

	def run_parser(self):

		self.parser = optparse.OptionParser()

		self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
		self.parser.add_option('--batch_size', type='int', default=20, dest='batch_size')
		self.parser.add_option('--max_epoch', type='int', default=200, dest='max_epoch')
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

		if(self.do_setup):

			with tf.variable_scope("model") as scope:

				if(self.model == "simple"):
					self.simple_model_setup()
				elif(self.model == "cnn"):
					self.cnn_model_setup()

		self.model_vars = tf.trainable_variables()
		for var in self.model_vars: print(var.name, var.get_shape())

		self.do_setup = False


	def loss_setup(self):

		self.loss = tf.reduce_mean(tf.squared_difference(self.pred, self.output_value))
		optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
		self.loss_optimizer = optimizer.minimize(self.loss)


	def train(self):

		self.model_setup()
		self.loss_setup()
		self.load_dataset()

		self.train_data, self.test_data =  divide_data(self.adj_close)

		tmp_size = self.train_data.shape
		self.input_data, self.output_data = pre_process(self.train_data, tmp_size[0], self.feature_size)
		self.output_data = np.reshape(self.output_data, [-1, 1])

		tmp_size = self.test_data.shape
		self.test_input_data, self.test_output_data = pre_process(self.test_data, tmp_size[0], self.feature_size)
		self.test_output_data = np.reshape(self.test_output_data, [-1, 1])

		input_data_size = self.input_data.shape[0]

		# print(self.input_data, self.output_data)

		# sys.exit()

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

			temp_test_input_feature = self.test_input_data[:self.batch_size]
			temp_test_output = self.test_output_data[:self.batch_size]
			# print(temp_test_input_feature, temp_test_output)
			# sys.exit()

			for epoch in range(0, self.max_epoch):


				for itr in range(0, (int)(input_data_size/self.batch_size)):
					
					temp_input_feature = self.input_data[itr*self.batch_size:(itr+1)*self.batch_size]
					temp_output = self.output_data[itr*self.batch_size:(itr+1)*self.batch_size]

					# print(temp_input_feature.shape, temp_output.shape)
					# sys.exit()

					_, temp_loss = sess.run([self.loss_optimizer, self.loss], 
						feed_dict={self.input_feature:temp_input_feature, self.output_value:temp_output})

					# if(itr%100 == 0):
					# 	print("In the epoch " + str(epoch) + " and the iteration " + str(itr) + " with a loss of " + str(temp_loss))

				temp_loss = sess.run([self.loss], 
						feed_dict={self.input_feature:temp_test_input_feature, self.output_value:temp_output})
				print("In the epoch " + str(epoch) + " with a loss of " + str(temp_loss))				


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


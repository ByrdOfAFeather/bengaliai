import tensorflow as tf
from tensorflow import nn


class InceptionNet:
	def __init__(self):
		self.pre_inception_filters = [
			tf.Variable(tf.random.normal(shape=[7, 7, 1, 64])),
			tf.Variable(tf.random.normal(shape=[1, 1, 64, 64])),
			tf.Variable(tf.random.normal(shape=[3, 3, 64, 192]))
		]

		self.inception_1_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 192, 64])),

			tf.Variable(tf.random.normal(shape=[1, 1, 192, 96])),
			tf.Variable(tf.random.normal(shape=[1, 1, 192, 16])),

			tf.Variable(tf.random.normal(shape=[3, 3, 96, 128])),
			tf.Variable(tf.random.normal(shape=[5, 5, 16, 32])),

			tf.Variable(tf.random.normal(shape=[1, 1, 192, 32]))
		]

		self.inception_2_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 256, 128])),

			tf.Variable(tf.random.normal(shape=[1, 1, 256, 128])),
			tf.Variable(tf.random.normal(shape=[1, 1, 256, 32])),

			tf.Variable(tf.random.normal(shape=[3, 3, 128, 192])),
			tf.Variable(tf.random.normal(shape=[5, 5, 32, 96])),

			tf.Variable(tf.random.normal(shape=[1, 1, 256, 64]))
		]

		self.inception_3_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 480, 192])),

			tf.Variable(tf.random.normal(shape=[1, 1, 480, 96])),
			tf.Variable(tf.random.normal(shape=[1, 1, 480, 16])),

			tf.Variable(tf.random.normal(shape=[3, 3, 96, 208])),
			tf.Variable(tf.random.normal(shape=[5, 5, 16, 48])),

			tf.Variable(tf.random.normal(shape=[1, 1, 480, 64]))
		]

		self.inception_4_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 512, 160])),

			tf.Variable(tf.random.normal(shape=[1, 1, 512, 112])),
			tf.Variable(tf.random.normal(shape=[1, 1, 512, 24])),

			tf.Variable(tf.random.normal(shape=[3, 3, 112, 224])),
			tf.Variable(tf.random.normal(shape=[5, 5, 24, 64])),

			tf.Variable(tf.random.normal(shape=[1, 1, 512, 64]))
		]

		self.inception_5_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 512, 128])),

			tf.Variable(tf.random.normal(shape=[1, 1, 512, 128])),
			tf.Variable(tf.random.normal(shape=[1, 1, 512, 24])),

			tf.Variable(tf.random.normal(shape=[3, 3, 128, 256])),
			tf.Variable(tf.random.normal(shape=[5, 5, 24, 64])),

			tf.Variable(tf.random.normal(shape=[1, 1, 512, 64]))
		]

		self.inception_6_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 512, 112])),

			tf.Variable(tf.random.normal(shape=[1, 1, 512, 144])),
			tf.Variable(tf.random.normal(shape=[1, 1, 512, 32])),

			tf.Variable(tf.random.normal(shape=[3, 3, 144, 288])),
			tf.Variable(tf.random.normal(shape=[5, 5, 32, 64])),

			tf.Variable(tf.random.normal(shape=[1, 1, 512, 64]))
		]

		self.inception_7_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 528, 256])),

			tf.Variable(tf.random.normal(shape=[1, 1, 528, 160])),
			tf.Variable(tf.random.normal(shape=[1, 1, 528, 32])),

			tf.Variable(tf.random.normal(shape=[3, 3, 160, 320])),
			tf.Variable(tf.random.normal(shape=[5, 5, 32, 128])),

			tf.Variable(tf.random.normal(shape=[1, 1, 528, 128]))
		]

		self.inception_8_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 832, 256])),

			tf.Variable(tf.random.normal(shape=[1, 1, 832, 160])),
			tf.Variable(tf.random.normal(shape=[1, 1, 832, 32])),

			tf.Variable(tf.random.normal(shape=[3, 3, 160, 320])),
			tf.Variable(tf.random.normal(shape=[5, 5, 32, 128])),

			tf.Variable(tf.random.normal(shape=[1, 1, 832, 128]))
		]

		self.inception_9_filters = [
			tf.Variable(tf.random.normal(shape=[1, 1, 832, 384])),

			tf.Variable(tf.random.normal(shape=[1, 1, 832, 192])),
			tf.Variable(tf.random.normal(shape=[1, 1, 832, 48])),

			tf.Variable(tf.random.normal(shape=[3, 3, 192, 384])),
			tf.Variable(tf.random.normal(shape=[5, 5, 48, 128])),

			tf.Variable(tf.random.normal(shape=[1, 1, 832, 128]))
		]

		self.linear_weights = tf.Variable(tf.random.normal(shape=[1024, 1]))
		self.linear_bias = tf.Variable(tf.random.normal(shape=[1024, 1]))

		self.optimizer = tf.optimizers.Adagrad()

	@staticmethod
	def _inception_layer(previous_layer, filters):
		unfiltered_1_x_1_conv = nn.conv2d(previous_layer, filters[0], strides=(1, 1), padding="SAME")

		stacked_conv_1_x_1_conv_1 = nn.conv2d(previous_layer, filters=filters[1], strides=(1, 1), padding="SAME")
		stacked_conv_1_x_1_conv_2 = nn.conv2d(previous_layer, filters=filters[2], strides=(1, 1), padding="SAME")

		stacked_conv_3_x_3_conv = nn.conv2d(stacked_conv_1_x_1_conv_1, filters=filters[3], strides=(1, 1),
		                                    padding="SAME")
		stacked_conv_5_x_5_conv = nn.conv2d(stacked_conv_1_x_1_conv_2, filters=filters[4], strides=(1, 1),
		                                    padding="SAME")

		pooled_layer = nn.max_pool2d(previous_layer, [3, 3], strides=(1, 1), padding="SAME")
		pooled_layer_conv = nn.conv2d(pooled_layer, filters=filters[5], strides=(1, 1), padding="SAME")

		return tf.concat([unfiltered_1_x_1_conv, stacked_conv_3_x_3_conv, stacked_conv_5_x_5_conv, pooled_layer_conv],
		                 axis=3)

	@staticmethod
	def _convolutional_layer(layer_input, layer_filter, stride):
		return nn.conv2d(layer_input, filters=layer_filter, strides=stride, padding="SAME")

	@staticmethod
	def _max_pool(input_layer, size, stride):
		return nn.max_pool2d(input_layer, size, strides=stride, padding="SAME")

	@staticmethod
	def _pre_inception_layer(input_data, filters):
		"""Takes in input of size (128, 128, 1)
		:param input_data:
		:param weights:
		:return:
		"""
		first_conv = InceptionNet._convolutional_layer(input_data, filters[0], (2, 2))
		first_pool = InceptionNet._max_pool(first_conv, (3, 3), (2, 2))

		second_conv = InceptionNet._convolutional_layer(first_pool, filters[1], (1, 1))
		third_conv = InceptionNet._convolutional_layer(second_conv, filters[2], (1, 1))
		second_pool = InceptionNet._max_pool(third_conv, (3, 3), (2, 2))

		return second_pool

	def forward_pass(self, input_data):
		pre_inception = InceptionNet._pre_inception_layer(input_data, self.pre_inception_filters)

		inception_1 = InceptionNet._inception_layer(pre_inception, self.inception_1_filters)
		inception_2 = InceptionNet._inception_layer(inception_1, self.inception_2_filters)

		post_inception_pool = InceptionNet._max_pool(inception_2, (3, 3), (2, 2))

		inception_3 = InceptionNet._inception_layer(post_inception_pool, self.inception_3_filters)
		inception_4 = InceptionNet._inception_layer(inception_3, self.inception_4_filters)
		inception_5 = InceptionNet._inception_layer(inception_4, self.inception_5_filters)
		inception_6 = InceptionNet._inception_layer(inception_5, self.inception_6_filters)
		inception_7 = InceptionNet._inception_layer(inception_6, self.inception_7_filters)

		post_inception_pool_2 = InceptionNet._max_pool(inception_7, (3, 3), (2, 2))

		inception_8 = InceptionNet._inception_layer(post_inception_pool_2, self.inception_8_filters)
		inception_9 = InceptionNet._inception_layer(inception_8, self.inception_9_filters)

		post_inception_pool_3 = nn.avg_pool(inception_9, (7, 7), strides=4, padding="SAME")

		flatten_layer = tf.reshape(tf.keras.backend.flatten(post_inception_pool_3), [input_data.shape[0], 1024])

		# TODO: Downsample into 168 space
		linear_layer = tf.matmul(tf.transpose(self.linear_weights),
		                         tf.transpose(flatten_layer)) + self.linear_bias
		return nn.softmax(linear_layer)

	@staticmethod
	def _calc_loss(actual, pred):
		return tf.reduce_sum(actual * tf.math.log(pred) + (1 - actual) * tf.math.log(1 - pred))

	def train(self, data_set, epochs):
		for i in range(0, epochs):
			image_set, labels = next(iter(data_set))
			print(image_set.shape)
			labels = tf.cast(labels, tf.float32)
			with tf.GradientTape() as tape:
				loss = self._calc_loss(labels, self.forward_pass(image_set))

			VARIABLES = [filterer for filters in [self.inception_1_filters,
			                                      self.inception_2_filters,
			                                      self.inception_3_filters,
			                                      self.inception_4_filters,
			                                      self.inception_5_filters,
			                                      self.inception_6_filters,
			                                      self.inception_7_filters,
			                                      self.inception_8_filters,
			                                      self.inception_9_filters,
			                                      ] for filterer in filters]
			VARIABLES.append(self.linear_weights)
			VARIABLES.append(self.linear_bias)

			gradients = tape.gradient(loss, VARIABLES)
			self.optimizer.apply_gradients(zip(gradients, VARIABLES))
			print(gradients)

import tensorflow as tf
from tensorflow.keras.layers import (Dense, BatchNormalization, Reshape,
	LeakyReLU, Conv2DTranspose, Conv2D, Flatten, Dropout, Activation, Input) 
import numpy as np 
import os
from collections import defaultdict 
from inspect import signature

class DCGAN:

	def __init__(self, image_size, gen_layer_filters, desc_layer_filters,
		noise_dim, epochs, batch_size, k_steps, g_lr=1e-4, d_lr=1e-4,
		early_learning=True, shuffle=True, save_fq=5):
		self.image_size = image_size
		self.gen_layer_filters = gen_layer_filters
		self.desc_layer_filters = desc_layer_filters
		self.noise_dim = noise_dim
		self.epochs = epochs
		self.batch_size = batch_size
		# recommended in the original paper (https://arxiv.org/abs/1406.2661)
		self.k_steps = k_steps
		# learning rates for the Generator and the discriminator
		self.g_lr, self.d_lr = g_lr, d_lr
		# Also recommended in the original paper for efficient learning for the Generator  
		# (https://arxiv.org/abs/1406.2661)
		self.early_learning = early_learning
		self.shuffle = shuffle
		self.save_fq = save_fq
		g = tf.Graph()
		with g.as_default():
			self.build_graph()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
		self.sess = tf.Session(graph=g)
		self.sess.run(init)


	def __repr__(self):
		fields = ', '.join([attr for attr in signature(self.__init__).parameters])
		reprObj = f'{type(self).__name__}({fields})'
		return reprObj


	def build_graph(self):
		with tf.variable_scope("Placeholders"):
			# placeholder for noise vector z
			self.z_g = tf.placeholder(tf.float32, shape=(None, self.noise_dim),
				name="noise_vect")
			# placeholder for real data
			self.real_inputs = tf.placeholder(tf.float32,
				shape=(None, *self.image_size), name='real_inputs')
		with tf.variable_scope("Network"):
			self.generated_images = self._Generator(self.z_g)
			real_output = self._Descriminator(self.real_inputs)
			fake_output = self._Descriminator(self.generated_images)
			self.gen_loss = self.Generator_loss(fake_output)
			self.desc_loss = self.descriminator_loss(real_output, fake_output)
			gen_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
				scope="Network/Generator")
			desc_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
				scope="Network/Descriminator")
			self.gen_optimizer = tf.train.AdamOptimizer(self.g_lr)\
			.minimize(self.gen_loss, var_list=gen_weights)
			self.desc_optimizer = tf.train.AdamOptimizer(self.d_lr)\
			.minimize(self.desc_loss, var_list=desc_weights)
		return


	def descriminator_loss(self, real_output, fake_output):
		loss = tf.math.log(real_output) + tf.math.log(1 - fake_output)
		loss = - tf.reduce_mean(loss)
		return loss


	def Generator_loss(self, fake_output):
		return tf.reduce_mean(tf.math.log(1 - fake_output))


	def _Generator(self, noise_vect):
		with tf.variable_scope("Generator"):
			x = noise_vect
			x = Dense(7*7*256, use_bias=False)(x)
			x = Reshape((7, 7, 256))(x)
			for filter in self.gen_layer_filters:
				x = BatchNormalization()(x)
				x = LeakyReLU()(x)
				x = Conv2DTranspose(filter, (5, 5),
					strides=(2, 2), padding='same', use_bias=False)(x)
			gen_output = Activation("tanh")(x)
		return gen_output


	def _Descriminator(self, inputs):
		with tf.variable_scope("Descriminator"):
			x = inputs
			for filter in self.desc_layer_filters:
				x = Conv2D(filter, (5, 5), strides=(2, 2), padding='same')(x)
				x = LeakyReLU()(x)
				x = Dropout(0.3)(x)
			x = Flatten()(x)
			desc_pred = Dense(1, activation='sigmoid')(x)
		return desc_pred


	def batch_generator(self, images):
		if self.shuffle:
			np.random.shuffle(images)
		nbr_obs = images.shape[0]
		for i in range(0, nbr_obs, self.batch_size):
			yield images[i: i + self.batch_size, :]


	def fit(self, images):
		losses = defaultdict(list)
		for i in range(self.epochs):
			yields_imgs = self.batch_generator(images)
			avg_desc_loss = 0
			for _, batch_img in zip(range(self.k_steps), yields_imgs):
				noise = np.random.normal(size=[self.batch_size, self.noise_dim])
				_, hist_desc_loss = self.sess.run([self.desc_optimizer,
					self.desc_loss], feed_dict={self.real_inputs: batch_img,
					self.z_g: noise})
				avg_desc_loss += hist_desc_loss

			noise = np.random.normal(size=[self.batch_size, self.noise_dim])
			_, hist_gen_loss = self.sess.run([self.gen_optimizer, self.gen_loss],
				feed_dict={self.z_g: noise})
			avg_desc_loss = avg_desc_loss / self.k_steps
			losses["GEN"].append(hist_gen_loss)
			losses["DESC"].append(avg_desc_loss)
			print(f"epoch: {i} | desc_loss: {avg_desc_loss} | gen_loss: {hist_gen_loss}")
			# save the model every save_fq=15 epochs
			if (i + 1) % self.save_fq == 0:
				self.save_model((i + 1))
				print('model saved')
		return losses


	def generate_img(self, noise):
		img = self.sess.run(self.generated_images, feed_dict={self.z_g: noise})
		return img


	def save_model(self, save_steps, save_path='./model/'):
		self.saver.save(self.sess, f'{save_path}{type(self).__name__}_model',
			global_step=save_steps)


	def load_model(self, save_step, load_path='./model/'):
		self.saver.restore(self.sess, f'{load_path}{type(self).__name__}_model-{save_steps}')


if __name__ == '__main__':
	import matplotlib.pyplot as plt 

	BATCH_SIZE = 256
	BUFFER_SIZE = 60000
	(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
	#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	image_size = train_images.shape[1:]
	gen_layer_filters, desc_layer_filters = [128, 64, 1], [64, 128]
	noise_dim = 100
	epochs, k_steps = 7, 1
	model = DCGAN(image_size, gen_layer_filters, desc_layer_filters,
		noise_dim, epochs, BATCH_SIZE, k_steps)
	# model.fit(train_images)
	# noise = np.random.normal(size=[1, noise_dim])
	# img = model.generate_img(noise)
	# plt.imshow(img[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
	# plt.show()
	print(model)
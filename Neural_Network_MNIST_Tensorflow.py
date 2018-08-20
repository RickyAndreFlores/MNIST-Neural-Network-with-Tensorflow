import tensorflow as tf
import keras.datasets.mnist as mnist

# Created by Ricky Flores

def data():
	# Import the MNIST dataset
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Give it a channels/depth dimension and return as a tensor
	X_train = tf.expand_dims(x_train, axis=3)
	X_test = tf.expand_dims(x_test, axis=3)

	# Create one hot tensors of y_train for use in softmax
	# outputs a one_hot vector whose length = num_classes for each example
	Y_train = tf.one_hot(y_train, 10)
	Y_test = tf.one_hot(y_test, 10)

	# Initialize summary tensor of x_train images for display
	tf.summary.image("x-train batch examples", X_train)

	return X_train, Y_train, X_test, Y_test


def accuracy_func(y_hat, Y):
	"""
	Calculate the accuracy of the model as a percentage
	Outputs result in decimal form

	"""
	# Calculate accuracy
	predictions = tf.nn.softmax(y_hat)
	correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
	accur = tf.reduce_mean(tf.cast(correct, "float"))

	# Create a summary of accuracy
	accur_sum = tf.summary.scalar("Accuracy", accur)

	return accur, accur_sum

def model(X, Y, alpha, t):
	"""
	The Neural Network [Forward prop]
	X - > FC -> FC -> FC -> FC->[Softmax->Output->Cross Entropy]


	Completes the forward and backwards pass for this neural network

	"""

	# Turn into floats
	Y = tf.cast(Y, dtype=tf.float32)
	X = tf.cast(X, dtype=tf.float32)

	# Flatten input
	a0 = tf.contrib.layers.flatten(X)

	# Hidden layer one, input a[0] = X (activation from layer 0), with 18 neurons
	a1 = tf.contrib.layers.fully_connected(a0, num_outputs=18, activation_fn=None)

	# Hidden layer 2, using a[1] as input
	a2 = tf.contrib.layers.fully_connected(a1, num_outputs=18, activation_fn=None)

	# Hidden layer 3, using a[2] as input
	a3 = tf.contrib.layers.fully_connected(	a2, num_outputs=18, activation_fn=None)

	# Output layer, using a[3] as input, and 10 neurons(=num_classes)
	y_hat = tf.contrib.layers.fully_connected(a3, num_outputs=10, activation_fn=None)


	if t == "train":
		# Computes the cross entropy for each example
		cross_e = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=Y)

		# Optimizer for regression/backward pass
		learn_ = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cross_e)

		accur, accur_sum = accuracy_func(y_hat, Y)

		return learn_, accur, accur_sum

	elif t == "test":
		return accuracy_func(y_hat, Y)


with tf.Session() as sess:


	# Initialize coordinator to help execute multi threading smoothly
	coord = tf.train.Coordinator()
	# Runs all queue runners for multi threading
	threads = tf.train.start_queue_runners(coord=coord)

	# Pre-process data
	x, y, x2, y2 = data()

	print("Additional information")
	print(x.shape, "x_train initial shape")
	print(y.shape, "y_train initial shapes")
	print("--------\n")

	# Define how many iterations you want to run
	epochs = 200

	# Tensors for running and training the model
	train, accuracy, accur_summ = model(x, y, .05, "train")

	# Tensors for test set
	accuracy_t, accur_summ_t = model(x2, y2, .05, "test")

	# Initialize tensor to merge all summaries
	merged = tf.summary.merge_all()
	# Initialize writer
	writer = tf.summary.FileWriter(f"/logs/", sess.graph)

	# Initialize all variables stated so far
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# Execute graph
	for i in range(epochs):

		# Run model
		___, ac, ac_summary = sess.run([train, accuracy, accur_summ])

		# Print accuracy
		print("Accuracy after epoch {}".format(i + 1), ac)

		# Add summary to writer for output with tensorboard
		writer.add_summary(ac_summary, i)

	# Run test set
	ac_t, ac_summ_t = sess.run([accuracy_t, accur_summ_t])

	# Print accuracy and add to tensorboard
	print("\nTest set accuracy", ac_t)
	writer.add_summary(ac_summ_t)


	coord.request_stop()
	coord.join(threads)
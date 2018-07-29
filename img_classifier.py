from scipy import misc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

total_iterations=0 #counter

def get_img_mat(file_name):
	#get to a working directory
	img_path='/Users/jameschee/Desktop/Programming/python/Fun/twitter/'+file_name #where I stored my image
	os.chdir(img_path)

	#get data and store in an array:img
	file_count = len([f for f in os.walk(".").next()[2] if f[-4:] == ".jpg"]) #number of .jpg files
	img = [None] * file_count #where I would store array of each img
	for i in range(file_count):
		try:
			file_name = str(i+1)+'.jpg'
			img[i] = misc.imread(file_name)
		except IOError:
			img[i] = None
	return np.array(img)

def train():
	#Gettin my HyperParameters ready
	#Convolutional Layer 1
	filter_size1 = 3
	num_filters1 = 32

	#Convolutional Layer 2
	filter_size2 = 3
	num_filters2 = 32

	#Convolutional Layer 3
	filter_size3 = 3
	num_filters3 = 32

	#Fully-connected layer
	fc_size = 48

	#number of color channels RGB=3
	num_channels = 3

	#image-size in squares
	img_size = 48

	#size flattened
	img_size_flat = img_size*img_size*num_channels

	#shape of image
	img_shape = (img_size,img_size)

	#class info
	classes = ['Trump','Obama']
	num_classes = len(classes)

	#batch size
	batch_size = 32

	#validation split
	validation_size = .16

	early_stopping = None

	###############################################################
	#defining placeholders
	x = tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
	x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])

	y_true = tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
	y_true_cls = tf.argmax(y_true,dimension=1)

	#conv_layer 1
	layer_conv1, weights_conv1 = new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=True)

	#conv_layer 2
	layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)

	#conv_layer 3
	layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,num_input_channels=num_filters2,filter_size=filter_size3,num_filters=num_filters3,use_pooling=True)

	#flattened out the layer
	layer_flat, num_features = flatten_layer(layer_conv3)

	#Fully connected layer 1
	layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_relu=True)

	#Fully connected layer 2
	layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_relu=False)

	#predicted value
	y_pred = tf.nn.softmax(layer_fc2)
	#argmax predicted value
	y_pred_cls = tf.argmax(y_pred,dimension=1)

	#really train
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

	#performance Measures
	correct_prediction = tf.equal(y_pred_cls,y_true_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	#run Session
	session = tf.Session()
	session.run(tf.initialize_all_variables())
	optimize(num_iterations=1,batch_size,session)


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

def optimize(num_iterations,train_batch_size,session):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,total_iterations + num_iterations):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)
        
        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
	return tf.Variable(tf.constant(0.05,shape=[length]))
def new_conv_layer(input, num_input_channels,filter_size,num_filters,use_pooling=True):
	#shape of filter weights
	shape = [filter_size,filter_size,num_input_channels,num_filters]
	weights = new_weights(shape = shape)
	biases = new_biases(length = num_filters)
	layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
	layer+=biases
	if use_pooling:
		layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	layer = tf.nn.relu(layer)
	return layer, weights
def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer,[-1,num_features])
	return layer_flat,num_features
def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):
	weights = new_weights(shape=[num_inputs,num_outputs])
	biases = new_biases(length=num_outputs)
	layer = tf.matmul(input,weights)+biases
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer


#Step1 Get the images into a matrix
trump_img = get_img_mat('trump_raw_img') #got rgb matrix of trump related image
obama_img = get_img_mat('obama_raw_img') #got rgb matrix of obama related image

print(trump_img[0].shape)
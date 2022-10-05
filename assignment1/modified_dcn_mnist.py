import os
import time

# Import Tensorflow and start a session
import tensorflow as tf

if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()   
    
# Load MNIST dataset
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# import tensorflow_datasets
# mnist = tensorflow_datasets.load('mnist')

sess = tf.InteractiveSession()

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initializer=tf.initializers.glorot_uniform()

    W = tf.Variable(initializer(shape))
    
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initializer=tf.initializers.glorot_uniform()
    
    b = tf.Variable(initializer(shape))
    
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    return h_max

def scalar_summary(para, para_name):
    tf.summary.scalar(para_name + '_mean', tf.reduce_mean(para))
    tf.summary.scalar(para_name + '_max', tf.reduce_max(para))
    tf.summary.scalar(para_name + '_min', tf.reduce_min(para))
    tf.summary.scalar(para_name + '_std', (tf.reduce_sum(para - tf.reduce_mean(para)))**2)
    tf.summary.histogram(para_name + '_hist', para)

def main():
    # Specify training parameters
    result_dir = './modified_results/' # directory where the results from the training are saved
    max_step = 5500 # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time() # start timing

    # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK
    
    # placeholders for input data and input labeles
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')

    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    
    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)    
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

    # setup training
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='CrossEntropyLoss')
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='test_accuracy')
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='train_accuracy')


    # Add a scalar summary for the snapshot loss and statistics of weights, bias, inputs, activations of each layer.
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)
    scalar_summary(W_conv1, 'conv1_weight')
    scalar_summary(b_conv1, 'conv1_bias')
    scalar_summary(x_image, 'conv1_input')
    scalar_summary(h_conv1, 'conv1_activation')    
    scalar_summary(W_conv2, 'conv2_weight')
    scalar_summary(b_conv2, 'conv2_bias')
    scalar_summary(h_pool1, 'conv2_input')
    scalar_summary(h_conv2, 'conv2_activation')    
    scalar_summary(W_fc1, 'fc_weight')
    scalar_summary(b_fc1, 'fc_bias')
    scalar_summary(h_pool2_flat, 'fc_input')
    scalar_summary(W_fc2, 'softmax_weight')
    scalar_summary(b_fc2, 'softmax_bias')
    scalar_summary(h_fc1_drop, 'softmax_input')
   
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    
    # Monitor test error
    summary_accuracy = tf.summary.merge([tf.summary.scalar('test_accuracy', test_accuracy), tf.summary.scalar('train_accuracy', test_accuracy)])

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for i in range(max_step):
        batch = mnist.train.next_batch(50) # make the data batch, which is used in the training iteration.
                                            # the batch size is 50
        if i%100 == 0:
            # output the training accuracy every 100 iterations
            train_accuracy_ = train_accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy_))

            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # run one train_step

    # print test error
    print("test accuracy %g"%test_accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    stop_time = time.time()
    print('The training takes %f second to finish'%(stop_time - start_time))

if __name__ == "__main__":
    main()


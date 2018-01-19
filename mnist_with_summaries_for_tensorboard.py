import argparse, sys, tempfile, tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model import tag_constants
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.python.client import timeline

"""
from tensorflow.contrib.learn.python.learn.utils import export

from tensorflow.contrib.meta_graph_transform import meta_graph_transform
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
"""
def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
  with tf.name_scope('conv1'):
    h_conv1 = tf.nn.relu(conv2d(x_image, weight_variable([5, 5, 1, 32])) + bias_variable([32]))
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
  with tf.name_scope('conv2'):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_variable([5, 5, 32, 64])) +  bias_variable([64]))
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
  with tf.name_scope('fc1'):
    h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 7 * 7 * 64]), weight_variable([7 * 7 * 64, 1024])) + bias_variable([1024]))
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  with tf.name_scope('fc2'):
    y_conv = tf.matmul(h_fc1_drop, weight_variable([1024, 10])) + bias_variable([10])
  return y_conv, keep_prob
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))
def main(_):
  mnist = input_data.read_data_sets('/tmp')
  print(mnist)
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.int64, [None])
  y_conv, keep_prob = deepnn(x)
  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  with tf.name_scope('accuracy'):
    correct_prediction = tf.cast(tf.equal(tf.argmax(y_conv, 1), y_), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  graph_location = './saved_graph/'#tempfile.mkdtemp() 
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  #saver = tf.train.Saver()
  #builder_saved_model = tf.saved_model_builder.SavedModelBuilder('./saved_model_dir/')

  batch_size=50
  total_epochs_to_train=2.0
  num_train_images=50000
  total_images_to_process=total_epochs_to_train*num_train_images
  total_batches_to_process=int(total_images_to_process/batch_size)
  #print('train dataset: ', mnist.train.output_types, mnist.train.output_shapes) print('validation dataset: ', mnist.validation.output_types, mnist.validation.output_shapes) print('test dataset: ', mnist.test.output_types, mnist.test.output_shapes)
  print('train dataset size: %d, val: %d, test: %d' % (mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples))
  print('total_images_to_process: %d, batch_size: %d, total_batches_to_process: %d, total_epochs: %g' % (total_images_to_process, batch_size, total_batches_to_process, total_epochs_to_train))
  saver = tf.train.Saver()
  
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()


  # create a summary for our cost and accuracy
  tf.scalar_summary("cost", cross_entropy)
  tf.scalar_summary("accuracy", accuracy)

  # merge all summaries into a single "operation" which we can execute in a session 
  summary_op = tf.merge_all_summaries()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
  # create log writer object
    writer = tf.train.SummaryWriter('./tensorboard', graph=tf.get_default_graph())
 
    for i in range(total_batches_to_process):
      batch = mnist.train.next_batch(batch_size)
      if i % 101 == 0:
        #loss, train_accuracy  = sess.run([cross_entropy, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        _, summary  = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
          writer.add_summary(summary, i)
        validation_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
        print('processed %d/%d images, batch %d/%d, epoch %g/%g, train acc: %g, val acc:%g, loss:%g' % (i*batch_size, total_images_to_process, i, total_batches_to_process, i*batch_size/num_train_images, total_epochs_to_train, train_accuracy, validation_accuracy, loss))
      if i % 1000 == 0:
        save_path = saver.save(sess, "./train_saver_1/")
        dir_to_export_model = saved_model_export_utils.get_timestamped_export_dir('./train_saver_2/')
        simple_save.simple_save(sess,  dir_to_export_model, {'x':x}, {'y_conv':y_conv})
        print("Model saved to file: %s, exported to %s" % (save_path, dir_to_export_model))
      if i % 917 == 0:
      
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, options=options, run_metadata=run_metadata)
        
      
         
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_step_%d.json' % i, 'a') as f:
          f.write(chrome_trace)

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
if __name__ == '__main__':
  tf.app.run(main=main)
  

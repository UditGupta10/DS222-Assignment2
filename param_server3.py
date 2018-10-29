import tensorflow as tf
import sys
import time

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
FLAGS = None

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

def main(_):
    f = open("mod_data",'r')
    tr_label = list()
    tr_features = list()
    for i in f.readlines():
            temp = i.split(' ')
            tr_label.append(int(temp[0]))
            tr_features.append(' '.join(temp[1:]))
    tr_label = np.array(tr_label)
    length_c = len(set(tr_label)) + 1

    f = open("mod_data_test",'r')
    ts_label = list()
    ts_features = list()
    for i in f.readlines():
            temp = i.split(' ')
            ts_label.append(int(temp[0]))
            ts_features.append(' '.join(temp[1:]))
    ts_label = np.array(ts_label)
    ls = Counter()
    for text in tr_features:
        for word in text.split(' '):
            ls[word.lower()]+=1
    for text in ts_features:
        for word in text.split(' '):
            ls[word.lower()]+=1

    tokens = len(ls)

    def get_word_2_index(ls):
        word2index = {}
        for i,word in enumerate(ls):
            word2index[word.lower()] = i
            
        return word2index

    word2index = get_word_2_index(ls)


    def get_batch(df,i,b_s,index):
        batches = []
        results = []
        tm_label = []
        tm_features = []
        size = int(len(tr_features) / no_of_workers)
        if df == 1:
            tm_label = tr_features[index * size : (index + 1) * size]
            tm_features = tr_label[index * size : (index + 1) * size]
        else:
            tm_label = ts_features
            tm_features = ts_label
        texts = tm_label[i*b_s:i*b_s+b_s]
        categories = tm_features[i*b_s:i*b_s+b_s]
        for text in texts:
            layer = np.zeros(tokens,dtype=float)
            for word in text.split(' '):
                layer[word2index[word.lower()]] += 1
                
            batches.append(layer)
            
        for category in categories:
            y = np.zeros((length_c),dtype=float)
            y[category] = 1.
            
            results.append(y)               
         
        return np.array(batches),np.array(results)



    # cluster specification

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    num_workers = len(worker_hosts)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    b_s = 100
    learning_rate = 0.5
    epochs = 10
    logs_path = "/tmp/mnist1/1"

    n_input = tokens 
    n_classes = length_c      

    if FLAGS.job_name == "ps":
      server.join()
    elif FLAGS.job_name == "worker":

      # Between-graph replication
      with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

        
        global_step = tf.get_variable('global_step', [],
                                    initializer = tf.constant_initializer(0),
                                    trainable = False)

        
        x = tf.placeholder(tf.float32, shape=[None, n_input], name="x-input")
        y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y-input")
      
        W1 = tf.Variable(tf.random_normal([n_input, n_classes]))
        b1 = tf.Variable(tf.zeros([n_classes]))
        out = tf.add(tf.matmul(x,W1),b1)
        y = tf.nn.softmax(out)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        grad_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.initialize_all_variables()


      
      begin_time = time.time()
      frq = 100
      with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, hooks=hooks) as sess:
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        start_time = time.time()
        for epoch in range(epochs):
          batch_count = int(len(tr_features)/(no_of_workers * b_s))
          print(batch_count)
          count = 0
          for i in range(batch_count):
            batch_x, batch_y =  get_batch(1,i,b_s,FLAGS.task_index)
           
            _, cost, step = sess.run(
                            [grad_op, cross_entropy, global_step],
                            feed_dict={x: batch_x, y_: batch_y})

            count += 1
            if count % frq == 0 or i+1 == batch_count:
              end_time = time.time() - start_time
              start_time = time.time()
              print("Step: %d," % (step+1),
                    " Epoch: %2d," % (epoch+1),
                    " Batch: %3d of %3d," % (i+1, batch_count),
                    " Cost: %.4f," % cost,
                    " AvgTime: %3.2fms" % float(end_time*1000/frq))
              count = 0
        batch_x, batch_y =  get_batch(2,0,len(ts_features)-1,FLAGS.task_index)
        print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y}))
        print("Total Time: %3.2fs" % float(time.time() - begin_time))

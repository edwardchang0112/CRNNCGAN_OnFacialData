#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import pandas as pd
import datetime
import csv
from Load_BatchData_txt import Read_batch_files_fromtxt

def readcsv(file_name, factor):
    dataset = pd.read_csv(file_name)
    #print (dataset[factor])
    return dataset[factor]

def readExcel(file_name, factor):
    dataset = pd.read_excel(file_name)
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]  # remvoe unnamed column names
    print("dataset = ", dataset)
    return dataset[factor]

def store_csv(Augment_Skin_data, window_size):
    with open('Augment_Skin_data'+str(datetime.datetime.now().strftime("%Y_%m_%d"))+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(Augment_Skin_data)

def moving_window(ts_data, window_size):
    #ts_data_win = np.zeros((len(ts_data)-window_size+1, window_size))
    ts_data_win = []
    for i in range(len(ts_data)-window_size+1):
        ts_data_win.append(ts_data[i:i+window_size])
    return ts_data_win

def data_rolling(ts_data, window_size):
    ts_data_arr = ts_data
    if window_size > len(ts_data_arr):
        ts_data_rol = moving_window(ts_data_arr, len(ts_data_arr))
    else:
        ts_data_rol = moving_window(ts_data_arr, window_size)
    return ts_data_rol

def date_diff(ts_date_win):
    date_diff_list = []
    #print("ts_date_win = ", ts_date_win)
    for date_row in ts_date_win:
        date_a = datetime.date(int(date_row[-1][:4]), int(date_row[-1][5:7]), int(date_row[-1][8:10]))
        date_diff_sublist = []
        for date in date_row:
            date_diff_sublist.append((date_a - datetime.date(int(date[:4]), int(date[5:7]), int(date[8:10]))).days)
        date_diff_list.append(date_diff_sublist[:-1])
    date_diff_list = np.stack((date_diff_list), axis=0)
    return date_diff_list

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))

def generator(z, y):
    input = tf.concat(axis=2, values=[z, y])
    input=tf.unstack(input,seq_size,1)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob)for _ in range(g_num_layers)])
    with tf.variable_scope("gen") as gen:
        res, states = tf.nn.static_rnn(lstm_cell, input,dtype=tf.float32)
        weights=tf.Variable(tf.random_normal([n_hidden, input_size]))
        biases=tf.Variable(tf.random_normal([input_size]))
        for i in range(len(res)):
            res[i]=tf.nn.sigmoid(tf.matmul(res[i], weights) + biases)
        g_params=[v for v in tf.global_variables() if v.name.startswith(gen.name)]
    with tf.name_scope("gen_params"):
        for param in g_params:
            variable_summaries(param)
    return res, g_params

def discriminator(x, x_generated):
    #input = tf.concat(axis=2, values=[x, y])
    input=tf.unstack(x, seq_size, 1)
    x_generated=list(x_generated)
    x_in = tf.concat([input, x_generated], 1)
    x_in=tf.unstack(x_in, seq_size, 0)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob) for _ in range(d_num_layers)])
    with tf.variable_scope("dis") as dis:
        weights=tf.Variable(tf.random_normal([n_hidden, 1]))
        biases=tf.Variable(tf.random_normal([1]))
        outputs, states = tf.nn.static_rnn(lstm_cell, x_in, dtype=tf.float32)
        res=tf.matmul(outputs[-1], weights) + biases
        y_data = tf.nn.sigmoid(tf.slice(res, [0, 0], [batch_size, -1], name=None))
        y_generated = tf.nn.sigmoid(tf.slice(res, [batch_size, 0], [-1, -1], name=None))
        d_params=[v for v in tf.global_variables() if v.name.startswith(dis.name)]
    with tf.name_scope("desc_params"):
        for param in d_params:
            variable_summaries(param)
    return y_data, y_generated, d_params


def sample_Z(k, m, n):
    noise_arr_all = np.random.normal(0., 0.1, size=[k, m, n])
    return noise_arr_all

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def nparrayToTensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

def save(saver, sess, logdir, step):
    model_name = 'model'
    checkpoint_path = os.path.join(logdir, model_name)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

if __name__ == '__main__':
    ''' Load some real data '''
    file_path = 'User_data/'
    txtfile_name = 'User_Files.txt'
    file_name_all = Read_batch_files_fromtxt(txtfile_name)

    user_data_win_all = []
    date_temp_humid_win_all = []
    window_size = 15
    for file_name in file_name_all:
        print ("Input_file_name = ", file_name)

        # Filter out those users' data that fewer than 15 days
        Hydration = (readExcel(file_path + file_name, 'Avg3_Hydration')).astype(np.float64)
        if len(Hydration) < 15:
            continue

        # load data from user files
        Oxygen = (readExcel(file_path + file_name, 'Avg3_Skinhealth')).astype(np.float64)
        Date_Diff = (readExcel(file_path+file_name, 'Day_Diff')).astype(np.float64)
        Temperature = (readExcel(file_path+file_name, 'Temperature')).astype(np.float64)
        Humidity = (readExcel(file_path+file_name, 'Humidity')).astype(np.float64)
        SkincareRatio = (readExcel(file_path+file_name, 'Skincare_Ratio')).astype(np.float64)
        Age = (readExcel(file_path+file_name, 'Age')).astype(np.float64)

        user_data = np.stack((Hydration, Oxygen), axis=1)
        user_data_win = data_rolling(user_data, window_size)
        #print("user_data_win = ", user_data_win)

        user_data_win = np.stack((user_data_win), axis=0)
        user_data_win = [user_data_win[k].reshape(1, -1)[0] for k in range(len(user_data_win))]

        user_data_win_all.append(user_data_win)
        print("user_data_win_all = ", user_data_win_all)

        date_temp_humid = np.stack((Date_Diff, Temperature, Humidity, SkincareRatio, Age), axis=1)
        date_temp_humid_win = data_rolling(date_temp_humid, window_size)

        date_temp_humid_list = []
        for i in range(len(date_temp_humid_win)):
            date_temp_humid_sub_list = []
            for j in range(len(date_temp_humid_win[i])):
                date_temp_humid_sub_sub = date_temp_humid_win[i][j] - np.stack(([date_temp_humid_win[i][0, 0], 0, 0, 0, 0]))
                date_temp_humid_sub_list.append(date_temp_humid_sub_sub)
                #print("date_temp_humid_sub_list = ", date_temp_humid_sub_list)
            date_temp_humid_list.append(date_temp_humid_sub_list)

        date_temp_humid_win = date_temp_humid_list
        date_temp_humid_win = np.stack((date_temp_humid_win), axis=0)
        date_temp_humid_win = [date_temp_humid_win[k].reshape(1, -1)[0] for k in range(len(date_temp_humid_win))]
        #date_temp_humid_win = np.vstack((date_temp_humid_win))
        date_temp_humid_win_all.append(date_temp_humid_win)
        #date_temp_humid_win_all = np.vstack((date_temp_humid_win_all))
        print("date_temp_humid_win_all = ", date_temp_humid_win_all)

    user_data_win_all = np.vstack((user_data_win_all))
    print("user_data_win_all = ", user_data_win_all)
    date_temp_humid_win_all = np.vstack((date_temp_humid_win_all))
    print("date_temp_humid_win_all = ", date_temp_humid_win_all)
    for i in range(len(user_data_win_all)):
        for j in range(len(user_data_win_all[i])):
            user_data_win_all[i][j] = user_data_win_all[i][j]/100


    user_data_win_all = user_data_win_all.reshape([-1, 15, 2])
    print("user_data_win_all = ", user_data_win_all)
    print("user_data_win_all.reshape([-1, 15, 2] = ", user_data_win_all.reshape([-1, 15, 2]))

    for i in range(len(date_temp_humid_win_all)):
        #print("date_temp_humid_win_all[i] = ", date_temp_humid_win_all[i])
        for j in range(len(date_temp_humid_win_all[i])):
            date_temp_humid_win_all[i][j] = date_temp_humid_win_all[i][j]/100
            #print("date_temp_humid_win_all[i][j] = ", date_temp_humid_win_all[i][j])

    date_temp_humid_win_all = date_temp_humid_win_all.reshape([-1, 15, 5])

    # model parameters setting
    mb_size = 10
    seq_size = 15 # for 15 days
    X_dim = 2 # input data (hydration and oxygen)
    y_dim = 5 # condition data (day_diff, temperature, humidity, skincareRatio, age)
    Z_dim = 10 # 10 dimension of a random input vector
    input_size = X_dim
    batch_size = 10
    n_hidden = 30
    g_num_layers = 10
    d_num_layers = 10
    X = tf.placeholder(tf.float32, shape=[None, seq_size, X_dim], name="X")
    y = tf.placeholder(tf.float32, shape=[None, seq_size, y_dim], name='y')
    Z = tf.placeholder(tf.float32, shape=[None, seq_size, Z_dim], name="Z")
    keep_prob = np.sum(0.7).astype(np.float32)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    G_sample, g_params = generator(Z, y)
    y_data, y_generated, d_params = discriminator(X, G_sample)

    D_loss = - (tf.log(y_data) + tf.log(1-y_generated))
    #D_loss = tf.log(y_data) + tf.log(1 - y_generated)
    #D_loss = (tf.log(abs(y_data - y_generated)))
    G_loss = - tf.log(y_generated)
    #G_loss = tf.log(y_generated)
    #D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    #D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    #D_loss = D_loss_real + D_loss_fake
    #G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    optimizer_g = tf.train.AdamOptimizer(0.0001)
    optimizer_d = tf.train.AdamOptimizer(0.00001)

    D_solver = optimizer_d.minimize(D_loss, var_list=d_params)
    G_solver = optimizer_g.minimize(G_loss, var_list=g_params)

    '''
    Sample_file_name = 'DateDiff_Weather.csv'
    Date_Diff = (readcsv(Sample_file_name, 'Date_Diff')).astype(np.float64)
    Temperature = (readcsv(Sample_file_name, 'Temperature')).astype(np.float64)
    Humidity = (readcsv(Sample_file_name, 'Humidity')).astype(np.float64)
    SkincareRatio = (readcsv(Sample_file_name, 'Skincare_Ratio')).astype(np.float64)
    Age = (readcsv(Sample_file_name, 'Age')).astype(np.float64)
    Sample_y_data = np.stack((Date_Diff, Temperature, Humidity, SkincareRatio, Age), axis=1)
    Sample_y_data = [Sample_y_data[i].reshape(1, -1) for i in range(len(Sample_y_data))]
    Sample_y_data = np.hstack((Sample_y_data))/100
    Sample_y_data = Sample_y_data.reshape([-1, 15, 5])
    print ("Sample_y_data = ", Sample_y_data)
    '''
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out_CRNNCGAN/'):
        os.makedirs('out_CRNNCGAN/')

    if not os.path.exists('snapshots_CRNNCGAN/'):
        os.makedirs('snapshots_CRNNCGAN/')

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)

    # model training
    i = 0
    mb_start_index = 0
    for it in range(300000):
        '''
        # Run samples to see the quality of generated results real time
        if it % 1000 == 0:
            print("it = ", it)
            n_sample = 10
            Z_sample = sample_Z(n_sample, seq_size, Z_dim)
            #print("Z_sample = ", Z_sample)
            y_sample = list(Sample_y_data)*n_sample
            #print("y_sample = ", y_sample)
            start_time = datetime.datetime.now()
            samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})
            end_time = datetime.datetime.now()
            print(end_time - start_time)
            store_csv(samples, window_size)
        '''
        # organize batch data
        if mb_start_index+mb_size >= len(user_data_win_all) - 1:
            X_mb = np.vstack((user_data_win_all[mb_start_index:], user_data_win_all[:mb_size-(len(user_data_win_all)-mb_start_index)]))
            y_mb = np.vstack((date_temp_humid_win_all[mb_start_index:], date_temp_humid_win_all[:mb_size - (len(date_temp_humid_win_all)-mb_start_index)]))

            np.random.shuffle(user_data_win_all)
            np.random.shuffle(date_temp_humid_win_all)
            mb_start_index = 0
        else:
            X_mb = user_data_win_all[mb_start_index:(mb_start_index+mb_size)]
            y_mb = date_temp_humid_win_all[mb_start_index:(mb_start_index+mb_size)]

        X_mb = X_mb.reshape([-1, 15, 2]).astype('float32')
        y_mb = y_mb.reshape([-1, 15, 5]).astype('float32')
        #print("X_mb = ", X_mb)
        #print("X_mb.shape = ", X_mb.shape)

        mb_start_index += mb_size

        Z_sample = sample_Z(mb_size, seq_size, Z_dim)

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})
        if it % 3 == 0:
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})

        if it % 1000 == 0:
            save(saver, sess, 'snapshots_RNNCGAN2/', it)
            print('Iter: {}'.format(it))
            print('G_loss_curr = ', G_loss_curr)
            print('D_loss_curr = ', D_loss_curr)

        #if it % 50 == 0:
            #tf.summary.scalar('D_loss', D_loss_curr)
            #tf.summary.scalar('G_loss', G_loss_curr)

            #file_writer.add_summary(D_loss_curr, it)

        #file_writer.close()

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
from skimage import color,filters

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output

def DecomNet_simple(input):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        up8 =  upsample_and_concat( conv3, conv2, 64, 128 , 'g_up_1')
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'g_up_2')
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        conv10=slim.conv2d(conv9,3,[1,1], rate=1, activation_fn=None, scope='g_conv10')
        R_out = tf.sigmoid(conv10)

        l_conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='l_conv1_2')
        l_conv3=tf.concat([l_conv2, conv9],3)
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        l_conv4=slim.conv2d(l_conv3,1,[1,1], rate=1, activation_fn=None,scope='l_conv1_4')
        L_out = tf.sigmoid(l_conv4)

    return R_out, L_out

def Restoration_net(input_r, input_i):
    with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
        input_all = tf.concat([input_r,input_i], 3)
        
        conv1=slim.conv2d(input_all,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_1')
        conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_2')
        pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv2_1')
        conv2=slim.conv2d(conv2,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv2_2')
        pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_1')
        conv3=slim.conv2d(conv3,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_2')
        pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

        conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_1')
        conv4=slim.conv2d(conv4,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_2')
        pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

        conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='de_conv5_1')
        conv5=slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='de_conv5_2')

        up6 =  upsample_and_concat( conv5, conv4, 256, 512, 'up_6')

        conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv6_1')
        conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, 128, 256, 'up_7'  )
        conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv7_1')
        conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, 64, 128, 'up_8' )
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv8_1')
        conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, 32, 64, 'up_9' )
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv9_1')
        conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv9_2')

        conv10=slim.conv2d(conv9,3,[3,3], rate=1, activation_fn=None, scope='de_conv10')
    
        out = tf.sigmoid(conv10)
        return out

def Illumination_adjust_net(input_i, input_ratio):
    with tf.variable_scope('Illumination_adjust_net', reuse=tf.AUTO_REUSE):
        input_all = tf.concat([input_i, input_ratio], 3)
        
        conv1=slim.conv2d(input_all,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_1')
        conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_2')
        conv3=slim.conv2d(conv2,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_3')
        conv4=slim.conv2d(conv3,1,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_4')

        L_enhance = tf.sigmoid(conv4)
    return L_enhance


class Model():
    def __init__(self):
        self.sess = tf.Session()

        self.input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
        self.input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
        self.input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
        input_high_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high_r')
        input_high_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')
        self.input_low_i_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')

        [R_decom, I_decom] = DecomNet_simple(self.input_decom)
        self.decom_output_R = R_decom
        self.decom_output_I = I_decom
        self.output_r = Restoration_net(self.input_low_r, self.input_low_i)
        self.output_i = Illumination_adjust_net(self.input_low_i, self.input_low_i_ratio)

        var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        var_adjust = [var for var in tf.trainable_variables() if 'Illumination_adjust_net' in var.name]
        var_restoration = [var for var in tf.trainable_variables() if 'Restoration_net' in var.name]

        self.saver_Decom = tf.train.Saver(var_list = var_Decom)
        self.saver_adjust = tf.train.Saver(var_list=var_adjust)
        self.saver_restoration = tf.train.Saver(var_list=var_restoration)

    def __call__(self, x):
        x = self.forward(x)
        return x

    def forward(self, x):
        h, w, _ = x.shape
        input_low_eval = np.expand_dims(x, axis=0)

        decom_r_low, decom_i_low = self.sess.run([self.decom_output_R, self.decom_output_I], feed_dict={self.input_decom: input_low_eval})
        
        restoration_r = self.sess.run(self.output_r, feed_dict={self.input_low_r: decom_r_low, self.input_low_i: decom_i_low})
        ### change the ratio to get different exposure level, the value can be 0-5.0
        ratio = 5.0
        i_low_data_ratio = np.ones([h, w])*(ratio)
        i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
        i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)
        adjust_i = self.sess.run(self.output_i, feed_dict={self.input_low_i: decom_i_low, self.input_low_i_ratio: i_low_ratio_expand2})

        #The restoration result can find more details from very dark regions, however, it will restore the very dark regions
        #with gray colors, we use the following operator to alleviate this weakness.  
        decom_r_sq = np.squeeze(decom_r_low)
        r_gray = color.rgb2gray(decom_r_sq)
        r_gray_gaussion = filters.gaussian(r_gray, 3)
        low_i =  np.minimum((r_gray_gaussion*2)**0.5,1)
        low_i_expand_0 = np.expand_dims(low_i, axis = 0)
        low_i_expand_3 = np.expand_dims(low_i_expand_0, axis = 3)
        result_denoise = restoration_r*low_i_expand_3
        fusion4 = result_denoise*adjust_i
        
        #fusion = restoration_r*adjust_i
        # fuse with the original input to avoid over-exposure
        x = decom_i_low*input_low_eval + (1-decom_i_low)*fusion4

        return x

    def load_weight(self, decom_weight_dir='weights/KinD/decom_net_train/',\
            adjust_weight_dir='weights/KinD/illumination_adjust_net_train/',\
            restoration_weight_dir='weights/KinD/Restoration_net_train/'):
        ckpt_pre = tf.train.get_checkpoint_state(decom_weight_dir)
        if ckpt_pre:
            print('loaded '+ckpt_pre.model_checkpoint_path)
            self.saver_Decom.restore(self.sess, ckpt_pre.model_checkpoint_path)
        else:
            print('No decomnet checkpoint!')

        ckpt_adjust = tf.train.get_checkpoint_state(adjust_weight_dir)
        if ckpt_adjust:
            print('loaded '+ckpt_adjust.model_checkpoint_path)
            self.saver_adjust.restore(self.sess, ckpt_adjust.model_checkpoint_path)
        else:
            print("No adjust pre model!")

        ckpt = tf.train.get_checkpoint_state(restoration_weight_dir)
        if ckpt:
            print('loaded '+ckpt.model_checkpoint_path)
            self.saver_restoration.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("No restoration pre model!")
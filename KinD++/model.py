import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
from skimage import color,filters


def illu_attention_3_M(input_feature, input_i, name):
    kernel_size = 3
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        concat = tf.layers.conv2d(input_i,
                                    filters=1,
                                    kernel_size=[kernel_size,kernel_size],
                                    strides=[1,1],
                                    padding="same",
                                    activation=None,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=False,
                                    name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat#, concat

def pool_upsamping_3_M(input_feature, level, training, name):
  if level == 1:
    with tf.variable_scope(name):
      pu_conv = slim.conv2d(input_feature, input_feature.get_shape()[-1], [3,3], 1, padding='SAME' ,scope='pu_conv')
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      pu_conv = tf.nn.relu(pu_conv)
      conv_up = pu_conv
  elif level == 2:
    with tf.variable_scope(name):
      pu_net = slim.max_pool2d(input_feature, [2,2], 2, padding='SAME', scope='pu_net')
      pu_conv = slim.conv2d(pu_net, input_feature.get_shape()[-1], [3,3], 1, padding='SAME' ,scope='pu_conv')
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      pu_conv = tf.nn.relu(pu_conv)
      conv_up = slim.conv2d_transpose(pu_conv, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up')
  elif level == 4:
    with tf.variable_scope(name):
      pu_net = slim.max_pool2d(input_feature, [4,4], 4, padding='SAME', scope='pu_net')
      pu_conv = slim.conv2d(pu_net, input_feature.get_shape()[-1], [1,1], 1, padding='SAME' ,scope='pu_conv')
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      pu_conv = tf.nn.relu(pu_conv)
      conv_up_1 = slim.conv2d_transpose(pu_conv, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up_1')
      conv_up = slim.conv2d_transpose(conv_up_1, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up')

  return conv_up

def Multi_Scale_Module_3_M(input_feature, training, name):
    
    Scale_1 = pool_upsamping_3_M(input_feature, 1, training, name=name+'pu1')
    Scale_2 = pool_upsamping_3_M(input_feature, 2, training, name=name+'pu2')
    Scale_4 = pool_upsamping_3_M(input_feature, 4, training, name=name+'pu4')
    
    res = tf.concat([input_feature, Scale_1, Scale_2, Scale_4], axis=3)
    multi_scale_feature = slim.conv2d(res, input_feature.shape[3], [1,1], 1, padding='SAME', scope=name+'multi_scale_feature')
    return multi_scale_feature

def msia_3_M(input_feature, input_i, name, training):
    spatial_attention_feature = illu_attention_3_M(input_feature, input_i, name)
    msia_feature = Multi_Scale_Module_3_M(spatial_attention_feature, training, name)
    return msia_feature

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

def DecomNet(input):
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


def Restoration_net(input_r, input_i, training = True):
    with tf.variable_scope('Denoise_Net', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input_r, 32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_1')
        conv1=slim.conv2d(conv1,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_2')
        msia_1 = msia_3_M(conv1, input_i, name='de_conv1', training=training)#, name='de_conv1_22')

        conv2=slim.conv2d(msia_1,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv2_1')
        conv2=slim.conv2d(conv2,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv2_2')
        msia_2 = msia_3_M(conv2, input_i, name='de_conv2', training=training)

        conv3=slim.conv2d(msia_2,512,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_1')
        conv3=slim.conv2d(conv3,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_2')
        msia_3 = msia_3_M(conv3, input_i, name='de_conv3', training=training)

        conv4=slim.conv2d(msia_3,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_1')
        conv4=slim.conv2d(conv4,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_2')
        msia_4 = msia_3_M(conv4, input_i, name='de_conv4', training=training)

        conv5=slim.conv2d(msia_4,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv5_1')
        conv10=slim.conv2d(conv5,3,[3,3], rate=1, activation_fn=None, scope='de_conv10')
        out = tf.sigmoid(conv10)
        return out

def Illumination_adjust_net(input_i, input_ratio):
    with tf.variable_scope('I_enhance_Net', reuse=tf.AUTO_REUSE):
        input_all = tf.concat([input_i, input_ratio], 3)
        
        conv1=slim.conv2d(input_all,32,[3,3], rate=1, activation_fn=lrelu,scope='conv_1')
        conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='conv_2')
        conv3=slim.conv2d(conv2,32,[3,3], rate=1, activation_fn=lrelu,scope='conv_3')
        conv4=slim.conv2d(conv3,1,[3,3], rate=1, activation_fn=lrelu,scope='conv_4')

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

        [R_decom, I_decom] = DecomNet(self.input_decom)
        self.decom_output_R = R_decom
        self.decom_output_I = I_decom
        self.output_r = Restoration_net(self.input_low_r, self.input_low_i)
        self.output_i = Illumination_adjust_net(self.input_low_i, self.input_low_i_ratio)

        var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]
        var_restoration = [var for var in tf.trainable_variables() if 'Denoise_Net' in var.name]
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_restoration += bn_moving_vars

        self.saver_Decom = tf.train.Saver(var_list = var_Decom)
        self.saver_adjust = tf.train.Saver(var_list=var_adjust)
        self.saver_restoration = tf.train.Saver(var_list=var_restoration)

    def __call__(self, x):
        x = self.forward(x)
        return x

    def forward(self, x, ratio=5.0, adjustment=False):
        h, w, _ = x.shape
        input_low_eval = np.expand_dims(x, axis=0)

        decom_r_low, decom_i_low = self.sess.run([self.decom_output_R, self.decom_output_I], feed_dict={self.input_decom: input_low_eval})
        
        restoration_r = self.sess.run(self.output_r, feed_dict={self.input_low_r: decom_r_low, self.input_low_i: decom_i_low})
        ### change the ratio to get different exposure level, the value can be 0-5.0
        i_low_data_ratio = np.ones([h, w])*(ratio)
        i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis=2)
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
        
        if adjustment:
            fusion = decom_i_low*input_low_eval + (1-decom_i_low)*fusion4
        else:
            fusion = decom_i_low*input_low_eval + (1-decom_i_low)*result_denoise
        x = fusion

        return x

    def load_weight(self, decom_weight_dir='weights/KinD++/decom_model/',\
            adjust_weight_dir='weights/KinD++/illu_model/',\
            restoration_weight_dir='weights/KinD++/restoration_model/'):
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
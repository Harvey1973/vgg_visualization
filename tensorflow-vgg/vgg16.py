import inspect
import os

import numpy as np
import tensorflow as tf
import time
import skimage
import skimage.io
import skimage.transform
from scipy.misc import imsave

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None,deconv_layer = 'pool1'):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print(self.data_dict.keys())
        self.net = {}
        print("npy file loaded")
        self.pool_mask = {}
        self.strides={"conv1_1":1,"conv1_2":1,"pool1":2,
                     "conv2_1":2,"conv2_2":2,"pool2":4,
                     "conv3_1":4,"conv3_2":4,"conv3_3":4,"pool3":8,
                     "conv4_1":8,"conv4_2":8,"conv4_3":8,"pool4":16,
                     "conv5_1":16,"conv5_2":16,"conv5_3":16,"pool5":32,}
        self.channels = {}
        self.channels[deconv_layer] = 64
        
        self.build()
        self.net["input_deconv"] = tf.placeholder(shape=[1,int(224/self.strides[deconv_layer]),int(224/self.strides[deconv_layer]),self.channels[deconv_layer]],dtype=tf.float32)
        self.net["output_deconv"] = self.backward(this_layer=deconv_layer,feature_map=self.net["input_deconv"])

    def build(self):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        self.net['input'] = tf.placeholder("float", [1, 224, 224, 3])
        start_time = time.time()
        print("build model started")
        rgb_scaled = self.net['input'] * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        tmp = tf.tile(self.pool1,[1,1,2,2])
        tmp = tf.reshape(tmp,self.conv1_2.shape)
        self.pool_mask['1'] = tf.cast(tf.greater_equal(self.conv1_2,tmp),dtype = tf.float32)

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        tmp = tf.tile(self.pool2,[1,1,2,2])
        tmp = tf.reshape(tmp,self.conv2_2.shape)
        self.pool_mask['2'] = tf.cast(tf.greater_equal(self.conv2_2,tmp),dtype = tf.float32)

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        tmp = tf.tile(self.pool3,[1,1,2,2])
        tmp = tf.reshape(tmp,self.conv3_3.shape)
        self.pool_mask['3'] = tf.cast(tf.greater_equal(self.conv3_3,tmp),dtype = tf.float32)

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        tmp = tf.tile(self.pool4,[1,1,2,2])
        tmp = tf.reshape(tmp,self.conv4_3.shape)
        self.pool_mask['4'] = tf.cast(tf.greater_equal(self.conv4_3,tmp),dtype = tf.float32)


        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        tmp = tf.tile(self.pool5,[1,1,2,2])
        tmp = tf.reshape(tmp,self.conv5_3.shape)
        self.pool_mask['5'] = tf.cast(tf.greater_equal(self.conv5_3,tmp),dtype = tf.float32)
        print(self.pool_mask['5'].shape)


        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        #self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def backward(self,this_layer = 'pool1', feature_map = None):
        layer_index = int(this_layer[4])
        if this_layer.startswith('pool'):
            if layer_index >=3 :
                last_layer = 'conv'+ str(layer_index)+'_3'
            else :
                last_layer = 'conv' + str(layer_index)+'_2'
            tmp = tf.tile(feature_map,[1,1,2,2])
            tmp = tf.reshape(tmp,self.pool_mask[str(layer_index)].shape)
            last_layer_feature = tmp*self.pool_mask[str(layer_index)]
            print('last layer is' + str(last_layer))
            #recursion 
            return self.backward(this_layer = last_layer, feature_map = last_layer_feature)
        if this_layer.startswith('conv'):
            if layer_index <=2 :
                num_filter = 2
            else :
                num_filter = 3
            
            for k in range(num_filter,0,-1):
                last_layer = 'conv'+ str(layer_index)+'_'+str(k)
                print('last year is + %s'%last_layer)
                relu = tf.nn.relu(feature_map)
                #bias = tf.nn.bias_add(relu,-1*se)
                last_layer_channel = len(self.data_dict[last_layer][0][0][0])
                output_shape = [1,int(224/self.strides[last_layer]),int(224/self.strides[last_layer]),last_layer_channel]
                print('output shape is %s' %str(output_shape))
                last_layer_feature = tf.nn.conv2d_transpose(relu,self.data_dict[last_layer][0],output_shape = output_shape,strides = [1,1,1,1],padding="SAME")
                feature_map = last_layer_feature
            if layer_index == 1:
                return last_layer_feature
            return self.backward(this_layer = 'pool%d'%(layer_index-1),feature_map = last_layer_feature)                




    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")





            

        

            
            




def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img



deconv_layer = 'pool1'
vgg = Vgg16()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch = load_image('test_data/tiger.jpeg')
batch = batch.reshape((1,224,224,3))

feed_dict = {vgg.net['input']: batch}
#vgg.build(images)
f_ = sess.run(vgg.pool1,feed_dict=feed_dict)


f = np.zeros([1,int(224/vgg.strides['pool1']),int(224/vgg.strides['pool1']),64])
max_9th_value = np.sort(f_[:,:,:,0]).flatten()[-9]
max_9th_mask = np.greater_equal(f_[:,:,:,0],max_9th_value).astype("int8")
f[:,:,:,0] = max_9th_mask * f_[:,:,:,0]
print(f.shape)
print(type(vgg.net['input_deconv']))
print(type(vgg.net['output_deconv']))
img_v = sess.run(vgg.net['output_deconv'],feed_dict={vgg.net['input']:batch,vgg.net['input_deconv']:f})
imsave('deconv_'+str(0)+'.png',img_v[0])
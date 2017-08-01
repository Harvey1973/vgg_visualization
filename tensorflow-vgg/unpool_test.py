import numpy as np 



a = np.array([[1,2,3,4,11,12,13],[5,6,7,8,15,16,17],[9,10,11,12,18,89,90],[13,14,15,16,66,54,12],[13,14,15,16,66,54,12],[13,14,15,16,66,54,12],[13,14,15,16,66,54,12]])


a = a.reshape((1,1,7,7))

final_result = []
height = a.shape[2]
width = a.shape[3]
channel = a.shape[1]
for k in range(channel):
    for i in range(height):
        a_ = a[0][k][i]
        a_ = np.tile(a_,(2,1))
        new_a = np.transpose(a_).flatten()
        new_a = np.tile(new_a,2)
        final_result = np.concatenate([final_result,new_a])
final_result=np.reshape(final_result,(14,14))
print(final_result)

b = np.arange(height*2*height*2)
b = np.reshape(b,(height*2,height*2))
mask =(final_result==b)
max_unpool = mask*b
print(max_unpool.shape)



    def max_unpool(self,layer,max_pooled_layer,feature_map_to_see,first_layer):
        final_result = []
        a = max_pooled_layer
        height = a.shape[2]

        width = a.shape[3]
        for i in range(height):
            a_ = a[0][feature_map_to_see][i]
            a_ = np.tile(a_,(2,1))
            new_a = np.transpose(a_).flatten()
            new_a = np.tile(new_a,2)
            final_result = np.concatenate([final_result,new_a])

        final_result=np.reshape(final_result,(height*2,height*2))

        if first_layer:
            mask =(final_result==layer[0][feature_map_to_see])
            max_unpool = mask*layer[0][feature_map_to_see]
        else :
            mask =(final_result==layer[0][feature_map_to_see])
            max_unpool = mask*layer
        return max_unpool

    def backward_2(self,shape_list,layer_list,previous_layer_list,num_conv_filters,sess):
        for i in range(len(layer_list)):
            if i == 0:
                transformed = np.transpose(layer_list[i],(0,3,1,2))
            else :
                transformed = sess.run(value_)
            transformed_conv = np.transpose(previous_layer_list[i],(0,3,1,2))
            pool = transformed
            conv = transformed_conv
            for k in range(64):
                unpool = self.max_unpool(layer = conv,max_pooled_layer=pool,feature_map_to_see=k,first_layer = True)
                shape_ = [[1,256,56,56],[1,128,112,112],[1,64,224,224]]
                shape_list = [[1,56,56,256],[1,112,112,64],[1,224,224,3],[1,224,224,3]]
                #print(shape_.shape)
                value_ = np.zeros(shape_[i])
                value_[0][0] = unpool
                value_ = np.transpose(value_,(0,2,3,1))
                value_ = tf.constant(value_,dtype=tf.float32)
                for t in range(num_conv_filters[i]):

                    filter_ = np.array(self.data_dict['conv'+str(3-i)+'_'+str(num_conv_filters[i]-t)][0])
                    print('filter name')
                    print('conv'+str(3-i)+'_'+str(num_conv_filters[i]-t))
                    shape_list[i][-1] = filter_.shape[-1]
                    print('filter has shape')
                    print(filter_.shape)
                    print('output shape is ')
                    print(shape_list[i])
                    print('input has shape')
                    print(value_.shape)
                    conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape = shape_list[i],strides = [1,1,1,1],padding="SAME",data_format="NHWC")
                
                value_ = conv_transposed
                final_ = sess.run(value_)
                
                imsave('deconv_'+str(k)+'.png',final_[0])

            return value_

    def visualize(self,name,layer,previous_layer,sess):

        if name == 'pool5':
            for e in range(10):
                transformed_conv_5 = np.transpose(previous_layer[0],(0,3,1,2))
                pool = np.transpose(layer[0],(0,3,1,2))


                conv5_3 = transformed_conv_5   
    
                unpool_5 = self.max_unpool(max_pooled_layer=pool,layer=conv5_3,feature_map_to_see = 0,first_layer= True)

                value_ = np.zeros((1,512,14,14))
                value_[0][0] = unpool_5
                value_ = np.transpose(value_,(0,2,3,1))
                value_ = tf.constant(value_,dtype=tf.float32)
                'pool5'
                for i in range(3):
                    filter_ = np.array(self.data_dict['conv5_'+str(3-i)][0])
                    filter_ = np.transpose(filter_,(0,1,2,3))
                    conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,14,14,512],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    value_ = conv_transposed

                
                
                'pool4'
                #transformed_4 = np.transpose(value_,(0,3,1,2))
                transformed_conv_4 = np.transpose(previous_layer[1],(0,3,1,2))
                pool_4 = sess.run(value_)
                conv4_3 = transformed_conv_4
 
                pool_4 = np.transpose(pool_4,(0,3,1,2))
                unpool_4 = self.max_unpool(layer = conv4_3,max_pooled_layer=pool_4,feature_map_to_see = 0, first_layer= True)
                value_ = np.zeros((1,512,28,28))
                value_[0][0] = unpool_4
                value_ = np.transpose(value_,(0,2,3,1))
                value_ = tf.constant(value_,dtype=tf.float32)
                #value_ = unpool_4
                
                for i in range(3):
                    filter_ = np.array(self.data_dict['conv4_'+str(3-i)][0])
                    filter_ = np.transpose(filter_,(0,1,2,3))
                    if i ==1:
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,28,28,512],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')

                    elif i == 0 :
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,28,28,512],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    else : 
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,28,28,256],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    value_ = conv_transposed
                'pool3'
                #transformed_4 = np.transpose(value_,(0,3,1,2))
                transformed_conv_3 = np.transpose(previous_layer[2],(0,3,1,2))
                pool_3 = sess.run(value_)
                conv3_3 = transformed_conv_3
                pool_3 = np.transpose(pool_3,(0,3,1,2))
                unpool_3 = self.max_unpool(layer = conv3_3,max_pooled_layer=pool_3,feature_map_to_see = 0, first_layer= True)
                value_ = np.zeros((1,256,56,56))
                value_[0][0] = unpool_3
                value_ = np.transpose(value_,(0,2,3,1))
                value_ = tf.constant(value_,dtype=tf.float32)
                #value_ = unpool_4
                
                for i in range(3):
                    filter_ = np.array(self.data_dict['conv3_'+str(3-i)][0])
                    filter_ = np.transpose(filter_,(0,1,2,3))
                    if i ==1:
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,56,56,256],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')

                    elif i == 0 :
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,56,56,256],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    else : 
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,56,56,128],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    value_ = conv_transposed
                'pool2'
                #transformed_4 = np.transpose(value_,(0,3,1,2))
                transformed_conv_2 = np.transpose(previous_layer[3],(0,3,1,2))
                pool_2 = sess.run(value_)
                conv2_2 = transformed_conv_2
                pool_2 = np.transpose(pool_2,(0,3,1,2))
                unpool_2 = self.max_unpool(layer = conv2_2,max_pooled_layer=pool_2,feature_map_to_see = 0, first_layer= True)
                value_ = np.zeros((1,128,112,112))
                value_[0][0] = unpool_2
                value_ = np.transpose(value_,(0,2,3,1))
                value_ = tf.constant(value_,dtype=tf.float32)
                
                for i in range(2):
                    filter_ = np.array(self.data_dict['conv2_'+str(2-i)][0])
                    filter_ = np.transpose(filter_,(0,1,2,3))
                    if i ==1:
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,112,112,64],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')

                    elif i == 0 :
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,112,112,128],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    else : 
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,56,56,128],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    value_ = conv_transposed
                'pool1'
                #transformed_4 = np.transpose(value_,(0,3,1,2))
                transformed_conv_1 = np.transpose(previous_layer[4],(0,3,1,2))
                pool_1 = sess.run(value_)

                conv1_2 = transformed_conv_1

                pool_1 = np.transpose(pool_1,(0,3,1,2))
                unpool_1 = self.max_unpool(layer = conv1_2,max_pooled_layer=pool_1,feature_map_to_see = 0, first_layer= True)
                value_ = np.zeros((1,64,224,224))
                value_[0][0] = unpool_1
                value_ = np.transpose(value_,(0,2,3,1))
                value_ = tf.constant(value_,dtype=tf.float32)

                for i in range(2):
                    filter_ = np.array(self.data_dict['conv1_'+str(2-i)][0])
                    filter_ = np.transpose(filter_,(0,1,2,3))
                    if i ==1:
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,224,224,3],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')

                    elif i == 0 :
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,224,224,64],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    else : 
                        conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,56,56,128],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                    value_ = conv_transposed
                value_ = sess.run(value_)

                imsave('deconv_'+str(e)+'.png',value_[0])
        elif name == 'pool1':
            shape_list = [[1,112,112,128],[1,224,224,64]]
            layer_list = [layer[2],layer[3],layer[4]]
            previous_layer_list = [previous_layer[2],previous_layer[3],previous_layer[4]]
            num_conv_filters = [3,2,2]
            self.backward(shape_list,layer_list,previous_layer_list,num_conv_filters,sess)
            
            '''
            for i in range(64):
                unpool_1 = self.max_unpool(previous_layer[0],layer[0],feature_map_to_see = i,first_layer= True)
                value_ = np.zeros((1,64,224,224))
                value_[0][0] = unpool_1
                value_ = np.transpose(value_,(0,2,3,1))
                value_ = tf.constant(value_,dtype=tf.float32)
            
                filter_ = np.array(self.data_dict['conv1_2'][0])
                #filter_ = np.transpose(filter_,(0,1,3,2))

                conv_transposed = tf.nn.conv2d_transpose(value_,filter_,output_shape =[1,224,224,64],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                filter_ = np.array(self.data_dict['conv1_1'][0])
                print(np.array(self.data_dict['conv1_1'][0]).shape)
                conv_transposed = tf.nn.conv2d_transpose(conv_transposed,filter_,output_shape =[1,224,224,3],strides = [1,1,1,1],padding = 'SAME',data_format ='NHWC')
                value_ = conv_transposed
            '''
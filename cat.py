import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import IPython

def combine_image(fg,fg2,bg,x,y,x2,y2,scale=1.0):
    shape = fg.shape
    width = shape[1]
    height = shape[0]
    scaled_width = int(width*scale+0.5)
    scaled_height = int(height*scale+0.5)
    bg_shape = bg.shape
    bg_width = bg_shape[1]
    bg_height = bg_shape[0]
    angle = 0.0
    #angle = -(minute-15)*360/60
    # matrix = cv2.getRotationMatrix2D((center_x,center_y),angle,scale)
    # fg_s = cv2.warpAffine(fg,matrix,(scaled_width,scaled_height))
    fg_s = cv2.resize(fg,(scaled_width,scaled_height))
    combined = np.copy(bg)
    to_put = combined[y:y+scaled_height,x:x + scaled_width]
    to_put[fg_s[:,:,0]>0] = fg_s[fg_s[:,:,0]>0]

    shape = fg2.shape
    width = shape[1]
    height = shape[0]
    scaled_width = int(width*scale+0.5)
    scaled_height = int(height*scale+0.5)
    fg_s = cv2.resize(fg2,(scaled_width,scaled_height))

    to_put = combined[y2:y2+scaled_height,x2:x2 + scaled_width]
    to_put[fg_s[:,:,0]>0] = fg_s[fg_s[:,:,0]>0]

    return combined
def generate_data(fg,fg2,bg,data_size):
    fg_shape = fg.shape
    bg_shape = bg.shape
    x_margin = bg_shape[1] - fg_shape[1]
    y_margin = bg_shape[0] - fg_shape[0]

    fg2_shape = fg2.shape
    x2_margin = bg_shape[1] - fg2_shape[1]
    y2_margin = bg_shape[0] - fg2_shape[0]

    # scales = np.random.random(size=[data_size])*0.4+0.8
    Xs = np.random.randint(0,x_margin,size=[data_size])
    Ys = np.random.randint(0,y_margin,size=[data_size])
    X2s = np.random.randint(0,x2_margin,size=[data_size])
    Y2s = np.random.randint(0,y2_margin,size=[data_size])    
    train_data = [combine_image(fg,fg2,bg,x,y,x2,y2) for x,y,x2,y2 in zip(Xs, Ys, X2s,Y2s)]
    target_data = [[x*1.0/bg_shape[1],y*1.0/bg_shape[0],x2*1.0/bg_shape[1],y2*1.0/bg_shape[0]] for x,y,x2,y2 in zip(Xs, Ys,X2s,Y2s)]
    return np.array(train_data),np.array(target_data)
def main(train):
    batch_size = 50
    cat=cv2.imread('cat.png')
    dog=cv2.imread('dog.png')
    background=cv2.imread('background.png')
    cat = cat[:,:,[2,1,0]]
    cat[cat[:,:,0]>245]=[0,0,0]
    dog = dog[:,:,[2,1,0]]
    dog[dog[:,:,0]>245]=[0,0,0]
    background = background[:,:,[2,1,0]]
    input = tf.placeholder(shape=[batch_size,224,224,3],dtype=tf.float32)
    target = tf.placeholder(dtype=tf.float32,shape=[batch_size,4])
    filter1_weights = tf.Variable(tf.truncated_normal(shape=[5,5,3,32],stddev=0.01))
    filter1_bias = tf.Variable(tf.zeros(shape=[32]))
    filter2_weights = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev=0.01))
    filter2_bias = tf.Variable(tf.zeros(shape=[64]))

    conv1 = tf.nn.conv2d(input,filter1_weights,strides=[1,2,2,1],padding="SAME")
    conv1 = tf.nn.bias_add(conv1,filter1_bias)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    conv1 = tf.nn.conv2d(conv1,filter2_weights,strides=[1,2,2,1],padding="SAME")
    conv1 = tf.nn.bias_add(conv1,filter2_bias)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    shape = conv1.get_shape().as_list()

    batch = shape[0]
    size = shape[1]*shape[2]*shape[3]

    flat = tf.reshape(conv1,[batch,size])

    fc1_weights = tf.Variable(tf.truncated_normal(shape=[size,128],stddev=0.01))
    fc1_bias = tf.Variable(tf.zeros(dtype=tf.float32,shape=[128]))

    fc1 = tf.matmul(flat,fc1_weights) + fc1_bias

    fc2_weights = tf.Variable(tf.truncated_normal(shape=[128,4],stddev=0.01))
    fc2_bias = tf.Variable(tf.zeros(dtype=tf.float32,shape=[4]))

    fc2 = tf.matmul(fc1,fc2_weights) + fc2_bias

    loss = tf.nn.l2_loss((fc2-target))/batch_size

    trainer = tf.train.GradientDescentOptimizer(0.001)

    step = trainer.minimize(loss)

    epoch = 200
   
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if(train):
            train_data_g, target_data_g = generate_data(cat,dog,background,batch_size*100)
            for i in range(epoch):
                for j in range(100):
                    train_data = train_data_g[j*batch_size:j*batch_size+batch_size]
                    target_data = target_data_g[j*batch_size:j*batch_size+batch_size]
                    [l,s] = sess.run([loss,step],feed_dict={input:train_data,target:target_data})
                    print "loss is " + str(l)
            saver.save(sess,'cat.model')
        else:
            saver.restore(sess,'cat.model')
            v,l = generate_data(cat,dog,background,batch_size)
            [o] = sess.run([fc2],feed_dict={input:v})
            offset = o*224
            plotimage(v[0],offset[0],cat.shape,dog.shape)

def plotimage(image,pos,shape1,shape2):
    width = shape1[1]
    height = shape1[0]
    width2 = shape2[1]
    height2 = shape2[0]
    ax=plt.subplot(111)
    ax.axis('off')
    box=plt.Rectangle((pos[0],pos[1]),width,height,fill=False,color='blue')
    box2=plt.Rectangle((pos[2],pos[3]),width2,height2,fill=False,color='green')
    ax.add_patch(box)
    ax.add_patch(box2)
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    main(False)



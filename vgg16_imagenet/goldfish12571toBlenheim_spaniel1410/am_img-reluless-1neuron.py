
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from scipy.spatial.distance import euclidean
from scipy.misc import imsave

# In[2]:


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        #self.readimg(fakefilename)
        self.convlayers()
        self.fc_layers()
        #self.probs = tf.nn.softmax(self.fc3l)
        #self.lossandgrad()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
    
    def convlayers(self):
        self.parameters = []
        self.convs = []
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1_before = out
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv1_1)

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2_before = out
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv1_2)

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv2_1)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2_before = out
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv2_2)

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        self.convs.append(self.pool2)

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv3_1)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv3_2)

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv3_3)

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')
        self.convs.append(self.pool3)

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv4_1)

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv4_2)

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv4_3)

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        self.convs.append(self.pool4)

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv5_1)

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv5_2)

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.convs.append(self.conv5_3)

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        self.convs.append(self.pool5)
    
    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1_before = fc1l
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))
    def lossandgrad(self,targetfeat):
        targetfeat_tf = tf.convert_to_tensor(targetfeat, np.float32)
        #self.ip = tf.reduce_sum(tf.multiply(targetfeat_tf, self.conv1_2))
        #norm1 = tf.norm(targetfeat_tf)
        #norm2 = tf.norm(self.conv1_2)
        #self.nm = tf.square(norm1-norm2)
        #self.cs = self.ip/norm1/norm2
        #self.posloss = tf.nn.l2_loss(feat1relu-layer_1_relu)
        #self.negloss = -tf.reduce_sum(tf.nn.relu(-layer_1))
        #self.loss = -self.ip+2*self.nm
        #self.grad = tf.gradients(self.loss,self.imgs)


# In[3]:


sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(imgs, 'vgg16_weights.npz', sess)


# In[4]:


datamat = np.load("../imagenet_%d.npy"%256,mmap_mode='r')
img1 = datamat[201]
img1.resize((256,256,3))
img1 = imresize(img1, (224, 224))
#imgid=270
#img1 = imread("../results_churchoutdoor/fake_%04d.png"%imgid)
#for j in range(1):
#    img1=np.mean(np.concatenate([img1[0::2,0::2,None], img1[0::2,1::2,None], img1[1::2,0::2,None], img1[1::2,1::2,None]], axis=2),axis=2)
img1.resize((1,224,224,3))
img1 = img1.astype(np.float32)
featfc1 = sess.run(vgg.fc1_before,feed_dict={vgg.imgs: img1})
featfc1_s = featfc1.copy()
featfc1_s[featfc1_s<0]=0
featfc1_s[featfc1_s>0]=1
featfc1_relu = sess.run(vgg.fc1,feed_dict={vgg.imgs: img1})

#img2 = np.load("../tmp/fake_0201.npz")["img"].astype(np.float32)#imread('../tmp/fake_0201_nn_intp.png', mode='RGB').astype(np.float32)
#vgg.lossandgrad(targetfeat)


# In[5]:


posloss = tf.nn.l2_loss(featfc1_relu-tf.multiply(vgg.fc1_before,featfc1_s))
tmp = (featfc1_s-1).nonzero()[1]
#print tmp.shape
neg_s = []
negloss=[]
loss_am=[]
grad = []
for i in range(5):
    neg_s.append(featfc1_s.copy()-1)
    neg_s[i][0,tmp[i]]=0
    negloss.append(-tf.reduce_sum(tf.slice(tf.multiply(vgg.fc1_before,featfc1_s-1),[0,tmp[i]],[1,1]))
                   +tf.reduce_sum(tf.abs(tf.multiply(vgg.fc1_before,neg_s[i]))))
    loss_am.append(100*posloss+10*negloss[i])
    grad.append(tf.gradients(loss_am[i], imgs))


# In[6]:


#img2 = img1.copy()[0]
img2 = np.reshape(datamat[20200],(256,256,3))
img2=imresize(img2,(224,224))
print img2.shape
img2 = img2.astype(np.float32)
#img2 = np.zeros((128,128,3),dtype=np.float32)





# In[8]:




# In[49]:
imgno = []
realgrad=[]
for i in range(5):
    imgno.append(img2.copy())
    realgrad.append(0)

# In[50]:


num_iter = 1000000
lr = 0.01



# In[ ]:

diffpairs=[(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
for i in range(num_iter):
    if i%2000==0:
        print "\n"
        print i
    for j in range(5):
        l1,neg1,l,g = sess.run([posloss,negloss[j],loss_am[j],grad[j]], feed_dict={vgg.imgs: [imgno[j]]})
        realgrad[j] = 0.99*realgrad[j]+0.01*g[0][0]
        imgno[j]-=lr*realgrad[j]
        imgno[j] = np.clip(imgno[j],0,255)
        if i%2000==0:
            imsave("results_pixel/samp_%d_%d.png"%(i,j),imgno[j].astype(np.uint8))
            print "positive loss No.%d"%j,l1
            print "negative loss No.%d"%j,neg1
            print "loss No.%d"%j,l
    if i%2000==0:
        print "two image diff",
        for diffpair in diffpairs:
            finaldiff = euclidean(imgno[diffpair[0]].flatten().astype(np.float64),imgno[diffpair[1]].flatten().astype(np.float64))
            print finaldiff,
        print "\n"
        






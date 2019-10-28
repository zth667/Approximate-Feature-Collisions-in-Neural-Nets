
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
#from imagenet_classes import class_names
from scipy.misc import imsave
from scipy.spatial.distance import euclidean


# In[27]:


class vgg16:
    def __init__(self,  paras, fakefilename,weights=None, sess=None):
        self.paras = tf.clip_by_value(paras,-1e37,1e37)
        self.init_paras = self.readimg(fakefilename)
        self.paras_sf = tf.nn.softmax(self.paras,0)
        self.imgs = tf.slice(tf.reduce_sum(tf.multiply(self.imgpack,self.paras_sf),0),[0,0,0],[224,224,3])
        self.imgs = tf.reshape(self.imgs,[1,224,224,3])
        self.convlayers()
        self.fc_layers()
        #self.probs = tf.nn.softmax(self.fc3l)
        #self.lossandgrad()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def readimg(self, filename):
        datamat = np.load("../../imagenet_%d.npy"%256,mmap_mode='r')
        filez = np.load(filename)
        finalidx = filez["finalidx"]
        finaldist = filez["finaldist"]
        finalpos = filez["finalpos"]
        #grid = filez["grid"]
        isz=28
        asz=56+2*isz
        hasz=asz/2.
        id_mat = np.reshape(finalidx[:,:1],[25,25,1])
        pos_mat = np.reshape(finalpos[:,:1,:],[25,25,1,2])
        imgpack = np.zeros((625*1,224,224,3),dtype=np.float32)
        paras = np.zeros((625*1,224,224,1),dtype=np.float32)
        cnt = 0
        for i in range(25):
            for j in range(25):
                for num in range(1):
                    tmpimg = np.zeros((672,672,3),dtype=np.float32)
                    tmppara = np.zeros((672,672,1),dtype=np.float32)
                    curimg = datamat[id_mat[i,j,num]]
                    curimg.resize(256,256,3)
                    curimg = imresize(curimg,(224,224))
                    posx,posy = (pos_mat[i,j,num]/12*10.5).astype(np.int)
                    stx = 224+i*7-posx
                    sty = 224+j*7-posy
                    tmpimg[stx:stx+224,sty:sty+224] = curimg
                    xs = [-1,1,0,0,-1,-1,1,1]
                    ys = [0,0,-1,1,-1,1,-1,1]
                    flips = [0,0,1,1,(0,1),(0,1),(0,1),(0,1)]
                    for k in range(8):
                        tx = stx+xs[k]*224
                        ty = sty+ys[k]*224
                        ex=tx+224
                        ey=ty+224
                        if isinstance(flips[k],tuple):

                            flippedimg = np.flip(curimg,0)
                            flippedimg = np.flip(curimg,1)
                        else:
                            flippedimg = np.flip(curimg,flips[k])
                        if tx<0:
                            flippedimg = flippedimg[0-tx:flippedimg.shape[0]]
                            tx=0
                        if ex>672:
                            flippedimg = flippedimg[0:flippedimg.shape[0]+672-ex]
                            ex=672
                        if ty<0:
                            flippedimg = flippedimg[:,0-ty:flippedimg.shape[1]]
                            ty=0
                        if ey>672:
                            flippedimg = flippedimg[:,0:flippedimg.shape[1]+672-ey]
                            ey=672
                        tmpimg[tx:ex,ty:ey] = flippedimg
                    mask = np.zeros((672,672,3),dtype=np.float32)
                    mask[224+i*7-isz:224+i*7-isz+asz,224+j*7-isz:224+j*7-isz+asz]=1
                    tmpimg = np.multiply(tmpimg,mask) 
                    imgpack[cnt] = tmpimg[224:448,224:448]
                    cntx=0
                    for kx in range(224+i*7-isz,224+i*7-isz+asz):
                        cnty=0
                        for ky in range(224+j*7-isz,224+j*7-isz+asz):
                            tmppara[kx,ky]=(1.-abs(cntx-hasz)/hasz)*(1.-abs(cnty-hasz)/hasz)
                            cnty+=1
                        cntx+=1
                    paras[cnt]=tmppara[224:448,224:448]
                    cnt+=1
        self.imgpack = tf.convert_to_tensor(imgpack,dtype=tf.float32)
        #print np.sum(paras[:,128:256,128:256],axis=0)==0
        paras = paras/(np.sum(paras,axis=0))
        return paras
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
            self.conv5_3before = out
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
    def lossandgrad(self,featfc1,featfc1_s,featfc1_relu,img1):
        #targetfeat_tf = tf.convert_to_tensor(targetfeat, np.float32)
        #self.ip = tf.reduce_sum(tf.multiply(featfc1, self.fc1before))
        #norm1 = tf.norm(targetfeat_tf)
        #norm2 = tf.norm(self.conv2_2)
        #self.nm = tf.norm(featfc1-self.conv2_2)#tf.square(norm1-norm2)
        #self.cs = self.ip/norm1/norm2
        k1=tf.constant([
                [1, -1]
            ], dtype=tf.float32)
        kernel = tf.reshape(k1,[1,2,1,1])
        hori_diff = tf.nn.conv2d(self.paras_sf,kernel,[1, 1, 1, 1], "VALID")
        k2=tf.constant([
                [1], [-1]
            ], dtype=tf.float32)
        kernel = tf.reshape(k1,[2,1,1,1])
        vert_diff = tf.nn.conv2d(self.paras_sf,kernel,[1, 1, 1, 1], "VALID")
        self.diffnorm = tf.reduce_sum(tf.abs(hori_diff))+tf.reduce_sum(tf.abs(vert_diff))
        self.posloss = tf.nn.l2_loss(featfc1_relu-tf.multiply(self.fc1_before,featfc1_s))
        tmp = (featfc1_s-1).nonzero()[1]
        neg_s = []
        self.negloss=[]
        self.loss_am=[]
        self.grad = []
        for i in range(5):
            neg_s.append(featfc1_s.copy()-1)
            neg_s[i][0,tmp[i]]=0
            self.negloss.append(-tf.reduce_sum(tf.slice(tf.multiply(self.fc1_before,featfc1_s-1),[0,tmp[i]],[1,1]))
                           +tf.reduce_sum(tf.abs(tf.multiply(self.fc1_before,neg_s[i]))))
            self.loss_am.append(1000*self.posloss+1*self.negloss[i]+100*self.diffnorm)
            self.grad.append(tf.gradients(self.loss_am[i],self.paras))
        


# In[28]:


imgid=270
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session()
paras = tf.placeholder(tf.float32, [625*1, 224, 224, 1])
vgg = vgg16(paras,'../../imgnet_32/imagenet_n07932039_798_overlap.npz', '../vgg16_weights.npz', sess)


# In[29]:



#img1 = datamat[201]
datamat = np.load("../../imagenet_%d.npy"%256,mmap_mode='r')
#img1 = imread("../imgnet_32/n01530575_6993.JPEG")
img1 = datamat[201]
img1.resize(256,256,3)
for j in range(0):
    img1=np.mean(np.concatenate([img1[0::2,0::2,None], img1[0::2,1::2,None], img1[1::2,0::2,None], img1[1::2,1::2,None]], axis=2),axis=2)
img1 = imresize(img1,(224,224))
img1.resize((1,224,224,3))
img1 = img1.astype(np.float32)
targetfeat = np.load("../n02971356_5502feats.npz")
featfc1 = targetfeat["featfc1"]
featfc1_s = targetfeat["featfc1_s"]
featfc1_relu = targetfeat["featfc1_relu"]
vgg.lossandgrad(featfc1,featfc1_s,featfc1_relu,img1)


# In[30]:



def invsf(target):
    for i in range(224):
        for j in range(224):
            target[:,i,j] = np.log(target[:,i,j])
            for k in range(625*1):
                if target[k,i,j]==-np.inf: target[k,i,j]==-100
    return target


# In[31]:
foldername="eggnogtocarton"
num_iter = 30000
target=np.load("../eggnoginit.npy")
target = np.clip(target,-100,1e37)
parano = []
realgrad=[]
imgno = [None,None,None,None,None]
st_iter = 0
for i in range(5):
    parano.append(target.copy())
    #parano.append(np.load("para_%d_%d.npy"%(st_iter,i)))
    realgrad.append(0.0)



# In[45]:


lr=0.001

# In[49]:

diffpairs=[(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

for i in range(st_iter,st_iter+num_iter):
    if i%500==0:
        print i
    if i==2200: lr=0.01
    if i==3200: lr=0.02
    if i==4700: lr=0.04
    if i==7000: lr=0.06
    if i==9000: lr=0.08
    if i==11000: lr=0.1
    if i==13000: lr=0.01
    for j in range(5):
        g = sess.run(vgg.grad[j], feed_dict={vgg.paras: parano[j]})
        realgrad[j] = 0.99*realgrad[j]+0.01*g[0]
        parano[j]-=lr*realgrad[j]
        if i%500==0:
            diffnm,l1,neg1,imgno[j] = sess.run([vgg.diffnorm,vgg.posloss,vgg.negloss[j],vgg.imgs], feed_dict={vgg.paras: parano[j]})
            imsave("./samp_%d_%d.png"%(i,j),imgno[j][0].astype(np.uint8))
            np.save("./para_%d_%d.npy"%(i,j),parano[j])
            print "No.%d"%j,l1,neg1,diffnm
    if i%500==0:
        print "two image diff",
        for diffpair in diffpairs:
            finaldiff = euclidean(imgno[diffpair[0]].flatten().astype(np.float64),imgno[diffpair[1]].flatten().astype(np.float64))
            print finaldiff,
        print "\n"




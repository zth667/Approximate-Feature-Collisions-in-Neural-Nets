'''
Code for Fast k-Nearest Neighbour Search via Prioritized DCI

This code implements the method described in our paper, which can be found at https://arxiv.org/abs/1703.00440

Copyright (C) 2017    Ke Li, Jitendra Malik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import print_function
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from dci import DCI
from scipy.ndimage import imread
from scipy.misc import imsave
import math
import time
import re
def gen_data(ambient_dim, intrinsic_dim, num_points):
    latent_data = 2 * np.random.rand(num_points, intrinsic_dim) - 1     # Uniformly distributed on [-1,1)
    transformation = 2 * np.random.rand(intrinsic_dim, ambient_dim) - 1
    data = np.dot(latent_data, transformation)
    return data     # num_points x ambient_dim
def get_img(ambient_dim, num_points, down):
    data = np.empty([num_points, ambient_dim])
    for i in range(num_points):
        filename="../church_128/img%08d.png"%i
        if os.path.isfile(filename):
            rawimg = imread(filename,flatten=False)
            rawimg = rawimg.astype(np.float64)
            rawimg=np.mean(np.concatenate([rawimg[0::2,0::2,None], rawimg[0::2,1::2,None], rawimg[1::2,0::2,None], rawimg[1::2,1::2,None]], axis=2),axis=2)
            rawimg=np.mean(np.concatenate([rawimg[0::2,0::2,None], rawimg[0::2,1::2,None], rawimg[1::2,0::2,None], rawimg[1::2,1::2,None]], axis=2),axis=2)
            data[i] = rawimg.flatten()
        else:
            raise IOError('Cannot locate image file', filename)
    return data
def getquery():
    minx=0
    miny=0
    maxx=128
    maxy=128
    ambient_dim=32*32*5
    numx=(max(minx+1,maxx-32+1)-minx+32-1)/32
    numy=(max(miny+1,maxy-32+1)-miny+32-1)/32
    print(numx,numy)
    data = np.empty([numx*numy, ambient_dim],dtype= np.float32)
    d=32
    st=0
    for j in range(miny,max(miny+1,maxy-32+1),32):
        for k in range(minx,max(minx+1,maxx-32+1),32):
            eyeimg = fakeimg_rp[j:j+d,k:k+d,:].flatten().astype(np.float32)
            data[st] = eyeimg/np.linalg.norm(eyeimg)*255
            st+=1
    return data
def gen_res(num_points, queries,idx, down, sz):
    from shutil import copyfile
    num = queries.shape[0]
    for i in range(num):
        directory = "../save_%d/query_%d"%(sz,i)
        if not os.path.exists(directory):
            os.makedirs(directory)
        src = "../results/fake_%04d.png"%i
        dest = "../save_%d/query_%d/target_raw.png"%(sz,i)
        copyfile(src,dest)
        rawimg = imread(src,flatten=False)
        for j in range(down):
            rawimg=np.mean(np.concatenate([rawimg[0::2,0::2,None], rawimg[0::2,1::2,None], rawimg[1::2,0::2,None], rawimg[1::2,1::2,None]], axis=2),axis=2)
        imsave("../save_%d/query_%d/target.png"%(sz,i), rawimg)
        cnt=0
        for k in idx[i]:
            src = "../church_128/img%08d.png"%k
            dest = "../save_%d/query_%d/near_%d_raw.png"%(sz,i,cnt)
            copyfile(src,dest)
            rawimg = imread(src,flatten=False)
            for j in range(down):
                rawimg=np.mean(np.concatenate([rawimg[0::2,0::2,None], rawimg[0::2,1::2,None], rawimg[1::2,0::2,None], rawimg[1::2,1::2,None]], axis=2),axis=2)
            imsave("../save_%d/query_%d/near_%d.png"%(sz,i,cnt), rawimg)
            cnt+=1
def setmodulename(s):
    regex = re.compile(r'\".+npy\"')
    with open("imgutil.py",'r') as ff:
        data_img = ff.readlines()
    for idx, line in enumerate(data_img):
        data_img[idx]=regex.sub(s, line)
    with open("imgutil.py",'w') as ff:
        ff.writelines(data_img)
def runknn(sz=4,num_points=202599,num_queries=1,num_comp_indices=2,num_simp_indices=7,num_outer_iterations=202599,
           max_num_candidates=5000,num_neighbours = 10, patch_size=5):
    num_levels = 3
    construction_field_of_view = 10
    construction_prop_to_retrieve = 0.02
    query_field_of_view = 12000
    query_prop_to_retrieve = 1.0
    dim = patch_size*patch_size*5
    #setmodulename('\"../church_npy/church_%d.npy\"'%sz)
    
    #data_and_queries = get_img(dim, num_points + num_queries)
    #data = np.load("../church_npy/church_%d.npy"%sz,mmap_mode='r')#get_img(dim, num_points)#np.copy(data_and_queries[:num_points,:])
    down = int(math.log(256/sz,2))
    
    dci_db = DCI(dim, num_comp_indices, num_simp_indices)
    st = time.time()
    print ("before adding")
    dci_db.add(num_points,"../church_npy/church_%d_vgg_12_5.npy"%sz,num_levels = num_levels, field_of_view = construction_field_of_view, prop_to_retrieve = construction_prop_to_retrieve,load_from_file=1)
    print ("construction time:", time.time()-st)
    imgid=270
    flip=0
    datamat = np.load("../church_npy/church_%d.npy"%128,mmap_mode='r')
    rawimg = imread("../results_churchoutdoor/fake_%04d.png"%imgid)
    
    if flip:
        rawimg=np.flip(rawimg,1)
    #im = Image.fromarray(target, 'RGB')
    for j in range(1):
        rawimg=np.mean(np.concatenate([rawimg[0::2,0::2,None], rawimg[0::2,1::2,None], rawimg[1::2,0::2,None], rawimg[1::2,1::2,None]], axis=2),axis=2)
    rawimg = rawimg.astype(np.uint8)
    if flip:
        fakeimgs = np.load("../church_npy/fakechurch_%d_vgg_12_flip.npy"%sz,mmap_mode='r')
    else:
        fakeimgs = np.load("../church_npy/fakechurch_%d_vgg_12.npy"%sz,mmap_mode='r')
    mularr=np.load("../church_npy/rpvec_64_5.npy")
    fakeimgs_rp=fakeimgs[imgid].copy()
    print (fakeimgs_rp.shape)
    fakeimg_rp=np.dot(fakeimgs_rp,mularr)
    del fakeimgs
    del mularr
    minx=0
    miny=0
    maxx=128
    maxy=128
    ambient_dim=32*32*5
    numx=(max(minx+1,maxx-32+1)-minx+32-1)/32
    numy=(max(miny+1,maxy-32+1)-miny+32-1)/32
    print(numx,numy)
    queries = np.empty([25*25, ambient_dim],dtype= np.float32)
    d=32
    st=0
    for j in range(miny,max(miny+1,maxy-32+1),4):
        for k in range(minx,max(minx+1,maxx-32+1),4):
            eyeimg = fakeimg_rp[j:j+d,k:k+d,:].flatten().astype(np.float32)
            queries[st] = eyeimg/np.linalg.norm(eyeimg)*255
            st+=1
    print(queries.shape,queries.dtype,st)
    st = time.time()
    num_neighbours = 200
    query_field_of_view = 10000 #11000
    query_prop_to_retrieve = 1.0
    nearest_neighbour_idx, nearest_neighbour_dists = dci_db.query(queries, num_neighbours, field_of_view = query_field_of_view, prop_to_retrieve = query_prop_to_retrieve, blind=False)
    print ("query time:", time.time()-st)
    finaldist = np.array(nearest_neighbour_dists)
    rawidx = np.array(nearest_neighbour_idx)
    print(rawidx.shape)
    finalidx = rawidx/(25*25)
    finalpos = np.empty([rawidx.shape[0],200,2],dtype= np.int)
    for i in range(rawidx.shape[0]):
        for j in range(200):
            offset = rawidx[i,j]%(25*25)
            finalpos[i,j,0] = offset/25*4
            finalpos[i,j,1] = offset%25*4
    np.savez("fake_%04d_all_overlap.npz"%imgid,finalidx = finalidx, finaldist = finaldist, finalpos = finalpos)
    
    
    
    
    
    
    
    
    
    
    
    
    return
    queries = get_query(dim, 1, down,sz,patch_size,patch_size)#data_and_queries[num_points:,:]
    print(queries.shape,queries.dtype)
    queries = queries[0:2]
    st = time.time()
    nearest_neighbour_idx, nearest_neighbour_dists = dci_db.query(queries, num_neighbours, field_of_view = query_field_of_view, prop_to_retrieve = query_prop_to_retrieve, blind=False)
    print ("query time:", time.time()-st)
    #gen_res(num_points, queries, nearest_neighbour_idx, down, sz)
    #resfile = "../save_%d/res.txt"%(sz)
    #f = open(resfile,'w')
    print(np.array(nearest_neighbour_idx)[:,0])
    print(np.array(nearest_neighbour_dists)[:,0])
    np.save("churchidx_18_v7.npy", np.array(nearest_neighbour_idx))
    np.save("churchdist_18_v7.npy",np.array(nearest_neighbour_dists))
    dci_db.clear()
    #print(num_candidates)
    #print(nearest_neighbour_idx,file=f)
    #print(nearest_neighbour_dists,file=f)
    #print(num_candidates,file=f)
    #print ("time elapsed:", time.time()-st)
    #f.close()
def main(*args):
    #72706752 34560000 38146752 45567947
    runknn(sz=128,num_points=78891875,num_queries=10,num_comp_indices=2,num_simp_indices=7,num_outer_iterations=202599,
           max_num_candidates=5000,num_neighbours = 50,patch_size=32)
    return

if __name__ == '__main__':
    st = time.time()
    main(*sys.argv[1:])
    print ("time elapsed:", time.time()-st)

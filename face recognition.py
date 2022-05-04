from keras.models import Sequential
from keras.layers import Conv2D,ZeroPadding2D,Activation,Input,concatenate
from keras.models import Model
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda,Flatten,Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

K.set_image_data_format('channels_first')

import time
import cv2
import os
import sys
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import fr_utils
from inception_blocks_v2 import *

np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(threshold=10000000)

#获取模型
FRmodel = faceRecoModel(input_shape=(3,96,96))

#打印模型的总参数数量
print("参数数量：" + str(FRmodel.count_params()))

#------------用于绘制模型细节，可选--------------#
plot_model(FRmodel, to_file='FRmodel.png')
SVG(model_to_dot(FRmodel).create(prog='dot', format='svg'))
#------------------------------------------------#

def triplet_loss(y_true,y_label,alpha=0.2):

    anchor,pos,neg=y_label[0],y_label[1],y_label[2]
    d_pos=tf.reduce_sum(tf.square(tf.subtract(anchor,pos)),axis=-1)
    d_neg=tf.reduce_sum(tf.square(tf.subtract(anchor,neg)),axis=-1)
    basic_loss=tf.add(tf.subtract(d_pos,d_neg),alpha)
    loss=tf.reduce_sum(tf.maximum(basic_loss,0))

    return loss


with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))

start_time=time.process_time()
FRmodel.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])
fr_utils.load_weights_from_FaceNet(FRmodel)
end_time=time.process_time()
time=end_time-start_time
print("执行了：" + str(int(time / 60)) + "分" + str(int(time%60)) + "秒")

database = {}
database["danielle"] = fr_utils.img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = fr_utils.img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = fr_utils.img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = fr_utils.img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = fr_utils.img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = fr_utils.img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = fr_utils.img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = fr_utils.img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = fr_utils.img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = fr_utils.img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = fr_utils.img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = fr_utils.img_to_encoding("images/arnaud.jpg", FRmodel)

def verify(image_path,identity,database,model):

    code=fr_utils.img_to_encoding(image_path,model)
    d=np.linalg.norm(code-database[identity])

    if d < 0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与" + str(identity) + "不符！")
        is_door_open = False

    return d, is_door_open

verify("images/camera_0.jpg","younes",database,FRmodel)

verify("images/camera_2.jpg", "kian", database, FRmodel)

def who(image_path,database,model):

    code=fr_utils.img_to_encoding(image_path,model)
    d_min=100

    for (name,db_enc) in database.items():
        d=np.linalg.norm(code-db_enc)
        if d<d_min:
            d_min=d
            identity=name

    if d_min > 0.7:
            print("抱歉，您的信息不在数据库中。")
    else:
            print("姓名" + str(identity) + "  差距：" + str(d_min))

    return d_min,identity

who("images/camera_0.jpg", database, FRmodel)









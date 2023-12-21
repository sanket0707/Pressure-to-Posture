

!python --version
!pip install --upgrade tensorflow
!pip install tensorflow==2.9.0

import tensorflow as tf
print(tf.__version__)
!python --version

import csv
import cv2
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import os

from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.eager.context import jit_compile_rewrite_enabled
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

data_log = {
    'export_dir': '/content/drive/MyDrive/PPIT/',
    'joints': {'Left ankle': 12,
                'Left elbow': 4,
                'Left hip': 8,
                'Left knee': 10,
                'Left shoulder': 2,
                'Left toe': 14,
                'Left wrist': 6,
                'Right ankle': 11,
                'Right elbow': 3,
                'Right hip': 7,
                'Right knee': 9,
                'Right shoulder': 1,
                'Right toe': 13,
                'Right wrist': 5,
              #  'Left heel':15,
              #  'Right heel':16

               }
}


frame_no = 7
after_frame = 3
batch_size = 32
shuffle_buffer_size = 1000
num_epochs = 2

def load_pickle_file(file_path):
    # Load the data from the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
    
def restack_electrode_data(pressure, location):
    frame_data = np.zeros((13,24))
    for i in range(frame_data.shape[0]):
      for j in range(frame_data.shape[1]):
        # i.   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        # loc  0,  , 1,  , 2,  , 3,  , 4,  ,  5,   ,  6
        # pres. , 0,  , 1,  , 2,  , 3,  , 4,   ,  5,


        # j.   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ..., 22, 23
        # loc  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ..., 22, 23
        # pres.0, 0, 1, 1, 2, 2, 3, 3, 4, 4,  5,  5,  6,      11, 11
        source_i = i//2
        source_j = j
        source_data = location
        if i%2 != 0: # pressure
          source_data = pressure
          source_j = j // 2
        frame_data[i,j] = source_data[source_i, source_j]
    return frame_data

def data_generator(file_paths, batch_size, frame=frame_no, after_frame=after_frame):
    for file_path in file_paths:
        # Load the data from the pickle file

        data = load_pickle_file(file_path)

        # Extract the input data and target data
        reshaped_data = []

        for i in range(len(data["location"])):
          pressure = data["pressure"][i]
          location = data["location"][i]
          reshaped_data.append(restack_electrode_data(pressure, location))
        data["reshaped_electrode"] = np.array(reshaped_data)

        # Prepare input and target data
        input_data = []
        for i in range(len(data["reshaped_electrode"]) - frame + 1 - after_frame):
          input_data.append(data["reshaped_electrode"][i:i+frame+after_frame])
        input_data = np.array(input_data)

                # Ensure that the input data has the expected shape
        #print("Input data shape before reshaping:", input_data.shape)


        #target_data = data["heatmap"]
        # generate target data with only a few keypoints
        target_data = np.stack((
                                data["keypoint"][frame-1:-after_frame, data_log['joints']['Left toe']-1,:],
                                 data["keypoint"][frame-1:-after_frame, data_log['joints']['Right toe']-1,:],
                                data["keypoint"][frame-1:-after_frame, data_log['joints']['Left ankle']-1,:],
                                 data["keypoint"][frame-1:-after_frame, data_log['joints']['Right ankle']-1,:],
                                data["keypoint"][frame-1:-after_frame, data_log['joints']['Left knee']-1,:],
                                 data["keypoint"][frame-1:-after_frame, data_log['joints']['Right knee']-1,:],
                                data["keypoint"][frame-1:-after_frame, data_log['joints']['Left hip']-1,:],
                                 data["keypoint"][frame-1:-after_frame, data_log['joints']['Right hip']-1,:],
                                data["keypoint"][frame-1:-after_frame, data_log['joints']['Left shoulder']-1,:],
                                 data["keypoint"][frame-1:-after_frame, data_log['joints']['Right shoulder']-1,:],
                                data["keypoint"][frame-1:-after_frame, data_log['joints']['Left elbow']-1,:],
                                 data["keypoint"][frame-1:-after_frame, data_log['joints']['Right elbow']-1,:],
                                data["keypoint"][frame-1:-after_frame, data_log['joints']['Left wrist']-1,:],
                                 data["keypoint"][frame-1:-after_frame, data_log['joints']['Right wrist']-1,:],
                                ),
                                axis=1)

        target_data = data["keypoint"][frame-1:-after_frame]
        target_data = target_data/32767


        # Ensure that the target data has the expected shape
        #print("Target data shape:", target_data.shape)

        # Yield the data in batches
        for i in range(0, (len(input_data) - frame - after_frame)//batch_size, batch_size):
          yield input_data[i*batch_size:(i+1)*batch_size], target_data[i*batch_size:(i+1)*batch_size]




DATA_Dir = "/content/drive/MyDrive/PPIT/20230626/"
print(os.listdir(DATA_Dir))

# Get a list of all files in the dataset directory
all_files = os.listdir(DATA_Dir)

# Filter files to keep only those ending with ".p"
pickle_files = [file for file in all_files if file.endswith(".p")]
pickle_files

# Split the files into training and validation sets (adjust as needed)
split_ratio = 0.8
num_files = len(pickle_files)
num_train_files = int(split_ratio * num_files)

# Training and validation file paths
train_file_paths = [os.path.join(DATA_Dir, file) for file in pickle_files[:num_train_files]]
val_file_paths = [os.path.join(DATA_Dir, file) for file in pickle_files[num_train_files:]]

# Print the training file paths
for path in train_file_paths:
  print(path)

# Create a TensorFlow dataset for the training set
train_dataset = tf.data.Dataset.from_generator(lambda: data_generator(train_file_paths, batch_size), output_types=(tf.float32, tf.float32))
train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size).repeat(num_epochs)

# Create a TensorFlow dataset for the validation set
val_dataset = tf.data.Dataset.from_generator(lambda: data_generator(val_file_paths, batch_size), output_types=(tf.float32, tf.float32))
val_dataset = val_dataset.shuffle(buffer_size=shuffle_buffer_size).repeat(num_epochs)

# Create a TensorFlow dataset for prediction and visualization on the validation set
visual_dataset_val = tf.data.Dataset.from_generator(lambda: data_generator(val_file_paths, batch_size), output_types=(tf.float32, tf.float32))

# Create a TensorFlow dataset for prediction and visualization on the training set (excluding the first file)
visual_dataset_train = tf.data.Dataset.from_generator(lambda: data_generator(train_file_paths[1:], batch_size), output_types=(tf.float32, tf.float32))


input_shape_train = next(iter(train_dataset))[0].shape
input_shape_val = next(iter(val_dataset))[0].shape

print("Shape of input data in train_dataset:", input_shape_train)
print("Shape of input data in val_dataset:", input_shape_val)

#model
def construct_model(keep_layer=1):
  dropout=0.3
  decoder_layer = [1024,512,256,128,64]
  InputLayer = tf.keras.layers.Input(batch_size=32, shape=(frame_no+after_frame, 13, 24))
  # layer_index = tf.keras.layers.Input(shape=(10, 10, 1, 9))
  print("check the tensor type in InputLayer", InputLayer.get_shape())

  ######################################################
  #
  #   ENCODER
  #
  #######################################################

  x = InputLayer
  print(x.shape)

  x = tf.expand_dims(x, axis=-1)
  print(x.shape)

  x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                            # kernel_initializer=tf.keras.initializers.Ones(),
                            # bias_initializer=tf.keras.initializers.Ones()
                            )(x)
  x = tf.keras.layers.BatchNormalization(axis=-1)(x)
  x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  print(x.shape)

  # 2nd layers from 32 to 64
  x = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same')(x)
  x = tf.keras.layers.BatchNormalization(axis=-1)(x)
  x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  print(x.shape)

  # 3rd layers from 64 to 128
  x = tf.keras.layers.Conv2D(128, (4,2), strides=(2,2))(x)
  x = tf.keras.layers.BatchNormalization(axis=-1)(x)
  x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  print(x.shape)

  # 3rd layers from 64 to 128
  x = x[:,:,0,:,:]
  print(x.shape)
  x = tf.keras.layers.Conv2D(128, (4,1), strides=(3,1))(x)
  x = tf.keras.layers.BatchNormalization(axis=-1)(x)
  x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  print(x.shape)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(5376)(x)
  x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  print(x.shape)
  x = tf.keras.layers.Reshape((14,3,128))(x)
  print(x.shape)


  # # ######################################################
  # #
  # # ##   DECODER
  # # #
  # # #######################################################

  x = tf.keras.layers.Conv2D(32, (2,2), strides=1, padding='same')(x)
  x = tf.keras.activations.tanh(x)
  x = tf.keras.layers.BatchNormalization(axis=-1)(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  print(x.shape)


  # 6th layer from 64 to 2, 2 is the number of joints in keypoints

  x = tf.keras.layers.Conv2D(1, (2,2), strides=1, padding='same')(x)
  x = tf.keras.activations.tanh(x)
  print(x.shape)
  OutputLayer = x

  return InputLayer, OutputLayer

construct_model()


#define loss
class heatmap_to_keypoint():
    """
    It creates  a keypoint (frame, joints, xyz) from a heatmap (frame, joints, 3Dheatmap_resolution)
    Output keypoint should be in the range of (-32767,32767) as the input in keypoint_to_heatmap
    The 3Dheatmap_resolution is (20, 20, 18) in the MIT model.
    """

    def __init__(self, axis_range=32767):
        self.axis_range = axis_range

    def __call__(self, heatmap, heatmap_shape=[20,20,18]):
        """
        heatmap is numpy array, it has a shape of (batch_size, joints, 3Dheatmap_resolution_in_xyz_axis)
        return data of size (frame, joints, xyz), data range is [0,1] before reverse_normalise,
        [-32767,32767) after reverse_normalise
        """
        y, x, z = [tf.linspace(0., 1., int(heatmap_shape[i]))
              for i in range(3)]
        self.pos_xyz = tf.meshgrid(x,y,z) # shape is [3, 20, 20, 18] here
        # heatmap = tf.reshape(heatmap, (*heatmap.shape[:-3], *heatmap_shape)) might not need it

        # heatmap shape is (batch_size, joints, 3Dheatmap_resolution_in_xyz_axis)

        eps = 1e-12
        expected_xyz = [
            tf.reduce_sum(tf.multiply([[self.pos_xyz[i]]], heatmap), axis=[-3,-2,-1])
            / (tf.reduce_sum(heatmap, axis=[-3,-2,-1]) + eps)
            for i in range(3)
        ] #The output shape is (3, batch_size, 16) where 16 is the joint number

        expected_xyz = tf.transpose(expected_xyz, perm=[1, 2, 0])

        expected_xyz = tf.convert_to_tensor(expected_xyz) # shape of (batch_size, 16, 3)
        xyz_float = self.reverse_normalise(expected_xyz) # shape of (batch_size, 16, 3)
        return xyz_float

    def reverse_normalise(self, data):
        """
        data has the shape of (batch_size, joints, xyz)
        """
        return data * self.axis_range * 2 - self.axis_range
        
        

heatmap2keypoint = heatmap_to_keypoint()
# keypoint2heatmap = keypoint_to_heatmap()
# link = tf.convert_to_tensor(data_log['link_limit'][0])
# limit = tf.convert_to_tensor(data_log['link_limit'][1])


def heatmap_loss(heatmap_GT, heatmap_pred, epsilon=1e-12):

    # heatmap_out = tf.where(heatmap_pred < epsilon, 0.0, heatmap_pred)
    loss = tf.reduce_mean(tf.square(heatmap_pred - heatmap_GT) * (heatmap_GT + 0.5) * 2) * 1000
    # loss = tf.reduce_mean(heatmap_pred, axis=(0,1,2))
    # loss = tf.reduce_mean(heatmap_pred)
    return loss*100

def keypoint_loss(keypoint_GT, keypoint_pred):
    keypoint_pred = tf.squeeze(keypoint_pred)
    keypoint_pred = tf.squeeze(keypoint_pred)
    loss = tf.reduce_mean(tf.square(keypoint_pred - keypoint_GT))
    return loss

reindex = {
    14: 0,
    13: 1,
    16: 2,
    15: 3,
    12: 4,
    11: 5
}

def link_loss(keypoint_pred):
    # link limit has the shape of (number of links, 2, 2), Axis 1 is linked joints. Axis 2 is the lower and upper limit of the link
    # The keypoint data has a shape of (time_frame, joint, xyz), time_frame could be 1
    #keypoint_pred has a shape of (batch_size, 16, 3)
    loss = []

    keypoint_pred = tf.squeeze(keypoint_pred)
    keypoint_pred = tf.squeeze(keypoint_pred)


    for link_limit in data_log['link_limit']:
        link, limit = link_limit
        try: # This to to avoid excess links
          distance = tf.norm(keypoint_pred[:, reindex[link[1]], :] - keypoint_pred[:, reindex[link[0]], :],
                              ord='euclidean', axis=0)
        except:
          continue
        # Calcuate how much is the predicted link away from the range. 3000 here is to calibrate
        #the maximum loss to be less than 10. This is to keep consistence with the MIT model

        distance = distance * 32767
        dist_diff = tf.where((distance > limit[0]) & (distance < limit[1]), 0.0, distance)
        loss_per_link = tf.reduce_mean(dist_diff)
        loss.append(loss_per_link)
    return tf.reduce_mean(loss)

def custom_loss(keypoint_GT, keypoint_pred):

    # keypoint_pred = tf.map_fn(lambda x:
    #                           tf.py_function(heatmap2keypoint, [x], tf.float32),
    #                           heatmap_pred)

    # keypoint_pred = heatmap2keypoint(heatmap_pred, heatmap_shape=heatmap_shape) # shape of (batch_size, joint, 3)
    loss1 = keypoint_loss(keypoint_GT, keypoint_pred)
    # print('loss1', heatmap_pred)
    # loss2 = link_loss(keypoint_pred)/3000/2
    # print('loss2', loss2.value)

    return loss1



#compile model
from keras.callbacks import LambdaCallback
import datetime

def callbacks():
  # Define the CSVLogger callback to save logs to a CSV file

  log_dir_csv = os.path.join(data_log['export_dir'], 'model', 'submodel', 'training_logs.csv')
  csv_logger = keras.callbacks.CSVLogger(log_dir_csv)

  # Create a TensorBoard callback and specify the log directory
  log_dir_ten = os.path.join(data_log['export_dir'], 'model', 'submodel', 'logs', 'fit')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_ten, histogram_freq=1)

  # Define a custom callback to print the value of a tensor
  class PrintTensorCallback(tf.keras.callbacks.Callback):
      def on_train_batch_end(self, batch, logs=None):
          # print('Value of loss:',  self.model.layers[24].output)
          return
  ################## Add weight callback

  checkpoint_dir = os.path.join(data_log['export_dir'], 'model', 'submodel', 'checkpoint{epoch:02d}-{val_loss:.2f}')

  log_dir1 = log_dir_ten+ '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir1, histogram_freq=1)

  # Define a custom callback to stop the epoch early
  early_stopping = tf.keras.callbacks.EarlyStopping(
                                                  monitor='val_loss',
                                                  min_delta=0,
                                                  patience=3,
                                                  verbose=0,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=False,
                                                  # start_from_epoch=0
                                              )


  # Train the model with the callback
  # heatmap_GT_tensor = tf.convert_to_tensor(heatmap_GT)
  # , callbacks=[csv_logger]
  return early_stopping, csv_logger,tensorboard_callback
  
%reload_ext tensorboard
%load_ext tensorboard

##### To start a new model
InputLayer, OutputLayer = construct_model()
model = keras.Model(InputLayer, OutputLayer)
model.summary()

# print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
# Compile your model with your custom loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                 decay=0.003,
                                                #  clipvalue=1,
                                                 ),
              loss=custom_loss)
######



# #### To start from an existing model
# model_path = os.path.join(data_log['export_dir'], 'model_20230616_P2', 'checkpoint02-6.52')
# model = tf.keras.models.load_model(model_path, custom_objects={"keypoint_loss": keypoint_loss,
#                                                                "optimizer":tf.keras.optimizers.Adam(learning_rate=0.0005,
#                                                  decay=0.003)})
# ####

# debug
tf.debugging.experimental.enable_dump_debug_info(os.path.join(data_log['export_dir'], 'model', 'submodel', 'tmp','tfdbg'))
#######


model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset,
          callbacks=[*callbacks()],
          batch_size=batch_size, verbose=1)

# Disable the debugger
tf.debugging.experimental.disable_dump_debug_info()

model_path = os.path.join(data_log['export_dir'], 'model', 'submodel', 'tensorflow')
try:
  os.mkdir(model_path)
except:
  None
model.save(model_path)


plot_model(model, to_file=model_path+'.png', show_shapes=True)

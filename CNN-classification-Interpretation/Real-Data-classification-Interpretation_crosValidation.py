import os
from keras.regularizers import l2
from keras import Sequential
from skimage import io,util
from scipy.misc import imresize, imsave
from sklearn.utils import shuffle
from keras.utils import np_utils
import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten,Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
batch_size = 2
num_epochs =20
import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import imp
import innvestigate.applications.imagenet
import innvestigate.utils as iutils
from keras.models import load_model
# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", "/home/sahar/innvestigate/examples/utils.py")
imgnetutils = imp.load_source("utils_imagenet", "/home/sahar/innvestigate/examples/utils_imagenet.py")

#address to save the model
save_model_path = 'Real_data_model_cross_validation.h5'
# address to mean of the interpretation of the groups interpretation
test_path = '/mean_train_images/'
# address to save the interpretatopn results
test_interpretation ='interpretations/'
if not os.path.exists(test_interpretation):
    os.mkdir(test_interpretation)
model_for_interpretation ='Real_Data_model_cross_validation.h5'
###################################
#
###################################
from glob import glob
import numpy as np
from skimage.color import rgb2gray
def read_data():



    listdir_1 = sorted(glob('/home/sahar/Documents/Real-Data-LCMS/Using-a-Spike-In-Experiment-to-Evaluate-Analysis-of-LC-MS-Data/costomize-images-2/spike-in/*'))
    listdir_2 = sorted(glob('/home/sahar/Documents/Real-Data-LCMS/Using-a-Spike-In-Experiment-to-Evaluate-Analysis-of-LC-MS-Data/costomize-images-2/spike-no/*'))
    y = np.ones(len(listdir_1))
    fileList = np.append(listdir_1, listdir_2)
    y  = np.append(y, np.zeros(len(listdir_2)))
    y   = np_utils.to_categorical(y, 2)
    return fileList, y
    # return shuffle(fileList, y, random_state=0)
def imageLoader(files, y, batch_size):
    L = len(files)
    #this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = np.ndarray((batch_size, int(image_h), image_w,1), dtype='float64')
            count = 0
            for x in files[batch_start:limit]:
                image = io.imread(x)
                image = imresize(image, (int(image_h), image_w, 1))
                #image = util.invert(image)
                #image= image.astype('float')/255
                #image = rgb2gray(image)
                image = image[:,:,np.newaxis]
                X[count] = np.array(image)
                count += 1
            Y = y[batch_start:limit]
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size
#########################
#
#########################
def get_model():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=(1, 1),
                     activation='relu', kernel_regularizer=l2(0.01), padding='same' ,
                     input_shape=(image_h, image_w,1) ))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.01), padding='same' , activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(2, (3, 3), padding='same' ,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(2 ,activation='softmax'))
    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy', metrics=['acc'])
    return model
#########################
#
#########################
#
#########################
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import skimage
data, label = read_data()
data, label  = shuffle(data, label)

acc_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits=10)#, shuffle=True)
# K-fold Cross Validation model evaluation
fold_no = 1
for train, val in kfold.split(data, label):
    model =  get_model()
    model_checkpoint = ModelCheckpoint('small-net-512', monitor ='val_loss', save_best_only = True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tb', histogram_freq=0, write_graph=True, write_images=True)
    #H = model.fit_generator(imageLoader(fileList[train],y[train],batch_size), steps_per_epoch=np.ceil( len(fileList[train])/batch_size ), epochs=2 )
    H = model.fit_generator(imageLoader(data[train],label[train],batch_size), steps_per_epoch=np.ceil( len(data[train])/batch_size ),validation_data=imageLoader(data[val],label[val], batch_size), validation_steps=np.ceil( len(data[train])/batch_size ) ,epochs=num_epochs )
    #acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Increase fold number
    # plot the training loss and accuracy
    #number of epochs
    N = num_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower right")
    plt.savefig('realData'+str(fold_no)+'_plot.png')
    model.save('realData_fold'+str(fold_no)+'_model'+'.h5')
    np.save('realData_fold'+str(fold_no)+'_'+'history.npy',H.history)
    fold_no = fold_no + 1
print('traning is done')


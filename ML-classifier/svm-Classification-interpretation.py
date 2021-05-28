import os
from keras.regularizers import l2
from keras import Sequential
from skimage import io,util
from sklearn.utils import shuffle
from keras.utils import np_utils
import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten,Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
image_h = 1500
image_w = 2500
batch_size = 1
num_epochs =10
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
save_model_path = 'model_cross_validation.h5'
# address to mean of the interpretation of the groups interpretation
test_path = '/mean_train_images/'

###################################
#
###################################
from glob import glob
import numpy as np
from skimage.color import rgb2gray
def read_data():

    listdir_1 = sorted(glob('/media/sahar/Data/Lc-MsData/20peptide_Simulation/1_2_out_Images_1500_2500_NoiseFiltering/TOPPAS_out/004-ImageCreator-out/*'))
    listdir_2 = sorted(glob('/media/sahar/Data/Lc-MsData/22peptide_Simulation/1_2_out_Images_1500_2500_NoiseFiltering/TOPPAS_out/004-ImageCreator-out/*'))
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
            X = np.ndarray((batch_size, image_h, image_w,1), dtype='float64')
            count = 0
            for x in files[batch_start:limit]:
                image = io.imread(x)
                import skimage
                image = skimage.transform.resize(image, (image_h, image_w))
                image = util.invert(image)
                image= image.astype('float')/255
                image = rgb2gray(image)
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

#########################
#
#########################
# #y_ture: image of diff, y_pred: interpretation
def ROI_function(diff, diff_trs, interpretation, interpretation_trs):
    #Convert the variable to values 0 and 1 with threshhold 0.5
    y_true = np.zeros((image_h,image_w), np.int)
    y_pred = np.zeros((image_h,image_w), np.int)
    y_true[diff > diff_trs]=1
    y_pred[interpretation>=interpretation_trs]=1
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    roi = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    # the common area over the interpretation or (the percantage of relevant peaks with respect to all seleceted peaks)
    r_  = (intersection + smooth)/(np.sum(y_pred_f)+smooth)
    # the common arae over true difference of two class
    r__ = (intersection + smooth)/(np.sum(y_true_f)+smooth)
    print(roi , r_, r__)
    return roi , r_, r__, y_pred, y_true
#########################
#
#########################
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import skimage
data, label = read_data()
data, label  = shuffle(data, label)
fileList, fileList_test, y, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

#Test differences prepration
mean_test_22= np.zeros((image_h, image_w,3),np.float)
mean_test_20 = np.zeros((image_h, image_w,3),np.float)
y_test_reverse_categorical = np.argmax(y_test, axis=1)
num_test_20pep = np.count_nonzero(y_test_reverse_categorical)
num_test_22pep = np.count_nonzero(y_test_reverse_categorical==0)

for N, file in enumerate(fileList_test):
    image = io.imread(file)
    #image = skimage.transform.resize(image, (image_h, image_w))
    image = util.invert(image)
    image= image.astype('float')/255
    #image = rgb2gray(image)
    if y_test_reverse_categorical[N] ==1:
        mean_test_20 = mean_test_20 + np.abs(image)/num_test_20pep
    else:
        mean_test_22 = mean_test_22 + np.abs(image)/num_test_22pep

io.imsave('mean_test_22.png', mean_test_22)
io.imsave('mean_test_20.png', mean_test_20)

test_image_dif_ = ( np.subtract(mean_test_22, mean_test_20) )
#remove the negetive parts where the 20 pep has affect
test_image_dif = np.abs(test_image_dif_)

#Find the first ith important areas of the test image diff
indices_diff_image = []
data = np.copy(test_image_dif)
for i in range(2000):
    ind = np.unravel_index(np.argmax(data, axis=None), data.shape)
    if (ind[0]>20 and ind[1]>4):
        data[ind[0]-20:ind[0]+20, ind[1]-3:ind[1]+3] =0
        indices_diff_image.append(ind)
    else:
        data[ind[0], ind[1]] =0

#Visualize the first ith important areas of the test image diff
data = np.copy(test_image_dif)
test_image_dif_importance = np.zeros((test_image_dif.shape))
for i in range(200):
    test_image_dif_importance [ indices_diff_image[i][0]-20:indices_diff_image[i][0]+20, \
    indices_diff_image[i][1]-3:indices_diff_image[i][1]+3 ] \
        = data[indices_diff_image[i][0]-20:indices_diff_image[i][0]+20, \
          indices_diff_image[i][1]-3:indices_diff_image[i][1]+3]

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
#def Train_model():
#global mean_test_20, mean_test_22, test_image_dif_importance
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5)#, shuffle=True)
# K-fold Cross Validation model evaluation
fold_no = 1
for train, val in kfold.split(fileList, y):
    model =  get_model()
    model_checkpoint = ModelCheckpoint('small-net-512', monitor ='val_loss', save_best_only = True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tb', histogram_freq=0, write_graph=True, write_images=True)
    #H = model.fit_generator(imageLoader(fileList[train],y[train],batch_size), steps_per_epoch=np.ceil( len(fileList[train])/batch_size ), epochs=2 )
    H = model.fit_generator(imageLoader(fileList[train],y[train],batch_size), steps_per_epoch=np.ceil( len(fileList[train])/batch_size ),validation_data=imageLoader(fileList[val],y[val], batch_size), validation_steps=np.ceil( len(fileList[train])/batch_size ) ,epochs=num_epochs )
    scores = model.evaluate_generator(imageLoader(fileList_test,y_test, batch_size), steps = np.ceil( len(fileList_test)/batch_size))
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
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
    plt.savefig(str(fold_no)+'_plot.png')
    model.save('fold'+str(fold_no)+'_model'+'.h5')
    np.save('fold'+str(fold_no)+'_'+'history.npy',H.history)
    method = "lrp.z"
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer(method, model_wo_softmax) # model without softmax output
    #Analyse
    mean_test_20_ = mean_test_20[:,:,np.newaxis]
    mean_test_20_= mean_test_20_[None,:,:,:]
    interpretation_mean_test_20 = analyzer.analyze(mean_test_20_)
    interpretation_mean_test_20 = interpretation_mean_test_20.sum(axis=np.argmax(np.asarray(interpretation_mean_test_20.shape) == 3))
    interpretation_mean_test_20/= np.max(np.abs(interpretation_mean_test_20))
    outputname = str(fold_no)+ str( method )+ 'interpretation_mean_test_20' +'.npy'
    np.save(outputname, interpretation_mean_test_20[:,:,0])

    #Analyse
    mean_test_22_ = mean_test_22[:,:,np.newaxis]
    mean_test_22_= mean_test_22_[None,:,:,:]
    interpretation_mean_test_22 = analyzer.analyze(mean_test_22_)
    interpretation_mean_test_22 = interpretation_mean_test_22.sum(axis=np.argmax(np.asarray(interpretation_mean_test_22.shape) == 3))
    interpretation_mean_test_22/= np.max(np.abs(interpretation_mean_test_22))
    outputname = str(fold_no)+ str( method )+ 'interpretation_mean_test_22' +'.npy'
    np.save(outputname, interpretation_mean_test_22[:,:,0])
    #feature selection on each fold

    interpretation_mean_test_22 = interpretation_mean_test_22[:,:,0]
    interpretation_mean_test_20 = interpretation_mean_test_20[:,:,0]

    indices_improved = []
    data_improved = np.copy(interpretation_mean_test_22)
    for i in range(10000):
        ind = np.unravel_index(np.argmax(data_improved, axis=None), data_improved.shape)
        #     if (ind[0]>20 and ind[1]>4) and (np.sum( Interpretation_20pep[ind[0]-20:ind[0]+20, ind[1]-3:ind[1]+3]) > -2):
        #         if np.sum( Interpretation_20pep[ind[0]-20:ind[0]+20, ind[1]-3:ind[1]+3]) >-1:
        if (ind[0]>20 and ind[1]>4 and interpretation_mean_test_20[ind[0], ind[1]] >= 0):
            data_improved[ind[0]-20:ind[0]+20, ind[1]-3:ind[1]+3] =0
            indices_improved.append(ind)
        else:
            data_improved[ind[0], ind[1]] =0


    data = np.copy(mean_test_22)
    Interpretation_importance = np.zeros((mean_test_22.shape))
    indices_ = indices_improved
    # for i in range(795):
    try:

        for i in range(200):
            if (indices_[i][0]>20 and indices_[i][1]>3):
                Interpretation_importance [ indices_[i][0]-20:indices_[i][0]+20, indices_[i][1]-3:indices_[i][1]+3] = \
                    data[indices_[i][0]-20:indices_[i][0]+20, indices_[i][1]-3:indices_[i][1]+3]
            else:
                Interpretation_importance [ indices_[i][0], indices_[i][1]] = \
                    data[indices_[i][0], indices_[i][1] ]

    except:
        pass

    # plt.imshow(Interpretation_importance, cmap="seismic", clim=(-0.1, 0.1))
    plt.imshow(test_image_dif_importance, cmap="seismic", clim=(-1, 1))

    # roi , r_, r__, y_pred_22, y_true_22 = ROI_function((test_image_dif_importance),0.03,Interpretation_importance,0.04)#the best threshhold
    roi , r_, r__, y_pred_22, y_true_22 = ROI_function(test_image_dif_importance,0.1,Interpretation_importance,0.1)#the best threshhold

    fold_no = fold_no + 1

print('traning is done')


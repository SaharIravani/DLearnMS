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
#address to save the model
save_model_path = 'model.h5'
# address to mean of the interpretation of the groups interpretation
test_path = '/mean_train_images/'
# address to save the interpretatopn results
test_interpretation ='interpretations/'
if not os.path.exists(test_interpretation):
    os.mkdir(test_interpretation)
model_for_interpretation ='model.h5'
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
                # image = imresize(image, (size_image, size_image, 3))
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
def get_model():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=(1, 1),
                     activation='relu', kernel_regularizer=l2(0.01), padding='same' ,
                     input_shape=(image_h, image_w,1) ))
    model.add(Dropout(0.3))
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
def Train_model():
    fileList, y = read_data()
    fileList, y  = shuffle(fileList, y)
    model =  get_model()
    model_checkpoint = ModelCheckpoint('small-net-512', monitor ='val_loss', save_best_only = True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tb', histogram_freq=0, write_graph=True, write_images=True)
    model.fit_generator(imageLoader(fileList,y,batch_size),steps_per_epoch=np.ceil( len(fileList)/batch_size ) , epochs=10 )
    model.save(save_model_path)
    print('traning is done')
#########################
#
#########################
#########################
#Interpretation
#########################
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
if keras.backend.backend() == "tensorflow":
    config = keras.backend.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.set_session(keras.backend.tf.Session(config=config))
def Interpret_model():
    image_path = test_path
    image_file= os.listdir(test_path)
    output_path =  test_interpretation
    model = load_model(model_for_interpretation)
    # Create model without trailing softmax
    method = "lrp.z"
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer(method, model_wo_softmax) # model without softmax output
    for x in image_file:
        image = io.imread(os.path.join(image_path, x))
        image.astype('float32') / 255
        image = rgb2gray(image)
        image = image[:,:,np.newaxis]
        image= image[None,:,:,:]
        a = analyzer.analyze(image)
        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
        a /= np.max(np.abs(a))
        plt.imshow(a[:,:,0], cmap="seismic", clim=(-1, 1))
        outputname = os.path.join(output_path , method +x)
        plt.savefig( outputname)
        outputname = os.path.join(output_path , method +x+'.npy')
        np.save(outputname, a[:,:,0])
        # io.imsave(outputname+'invert.png',  util.invert(a[:,:,0]))
    # return a
    print('interpretation is done')
#########################
#
#########################

def Kernel_Visualization():
    model = load_model(model_for_interpretation)
    model.summary()
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
    # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
    # retrieve weights from the third hidden layer
    filters, biases = model.layers[2].get_weights()
    # retrieve weights from the second hidden layer
    # filters, biases = model.layers[2].get_weights()
    # retrieve weights from the first hidden layer
    # filters, biases = model.layers[0].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 32, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        #Range(number of channels of each filter)
        for j in range(32):
            # ix %=8
            # specify subplot and turn of axis
            ax = plt.subplot(32,32, ix )
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()
    print('Kernel visualization is done!')
#########################
#
#########################
def featureMap_Visualization():
    model = load_model(model_for_interpretation)
    model.summary()
    image = io.imread('.../mean_trainImage_20pep.png')
    image.astype('float32') / 255
    image = rgb2gray(image)
    image = image[:,:,np.newaxis]
    image= image[None,:,:,:]
    model = Model(inputs=model.inputs, outputs=model.layers[12].output)
    feature_maps = model.predict(image)
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(2):
        for _ in range(1):
            # specify subplot and turn of axis
            ax = plt.subplot(2,1, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[ 0,:, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

if __name__== '__main__':
    Train_model()
    Interpret_model()
    # Kernel_Visualization()
    # featureMap_Visualization()

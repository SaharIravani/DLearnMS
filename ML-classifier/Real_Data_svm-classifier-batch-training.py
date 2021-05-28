import os
from skimage import io,util
from sklearn.utils import shuffle
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import io
import matplotlib.pyplot
import matplotlib.pyplot as plt
import imp
from glob import glob
import numpy as np
from skimage.color import rgb2gray
from skimage import io, transform, util
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
#address to save the model
save_model_path = 'model_cross_validation.h5'
# address to mean of the interpretation of the groups interpretation
test_path = '/mean_train_images/'
batch_size = 1
num_epochs =10
image_h = int(1142)
image_w = int(2000)
def read_data():
    listdir_1 = sorted(glob('/home/sahar/Documents/Real-Data-LCMS/Using-a-Spike-In-Experiment-to-Evaluate-Analysis-of-LC-MS-Data/costomize-images-2/spike-in/*'))
    listdir_2 = sorted(glob('/home/sahar/Documents/Real-Data-LCMS/Using-a-Spike-In-Experiment-to-Evaluate-Analysis-of-LC-MS-Data/costomize-images-2/spike-no/*'))
    y = np.ones(len(listdir_1))
    fileList = np.append(listdir_1, listdir_2)
    y  = np.append(y, np.zeros(len(listdir_2)))
    return fileList, y

fileList, labels= read_data()
#data, label  = shuffle(data, label)

data = np.ndarray((np.size(fileList), image_h*image_w), dtype='float64')
count = 0
for x in fileList:
    image = io.imread(x)
    import skimage
    image = transform.resize(image, (image_h, image_w))
    image = util.invert(image)
    #image= image.astype('float')/255
    image = rgb2gray(image)
    image = image.reshape(image.shape[0]*image.shape[1]).T
    data[count] = np.array(image)
    count += 1



#########################
#
#########################

#########################
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import skimage
import skimage.transform
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
data, labels = shuffle(data, labels)
labels_binary = np.copy(labels)
kf = KFold(n_splits=10 ,shuffle=True)
###############################################
# Classify whole data with  Random Forest
###############################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
c1=1
y1_Best_data = np.array([])
yhat1_Best_data = np.array([])
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #clf_Rf = RandomForestClassifier(max_depth=None, random_state=0)
    clf_Rf = DecisionTreeClassifier(max_depth=10)
    clf_Rf.fit(X_train, y_train)
    #print('Spars Data Accuracy for fold:',c3,clf3.score(X_test, y_test))
    c1 +=1
    y1_Best_data = np.concatenate([y1_Best_data, y_test])
    yhat1 = clf_Rf.predict(X_test)
    yhat1_Best_data = np.concatenate([yhat1_Best_data, yhat1])
    #metrics.plot_roc_curve(clf_Rf, X_test, y_test )
    #plt.show()

CM1 = confusion_matrix(y1_Best_data, yhat1_Best_data)
CM1 = CM1.astype(float)
TN1 = CM1[0][0]
FN1 = CM1[1][0]
TP1 = CM1[1][1]
FP1 = CM1[0][1]
Sen1 = np.divide(TP1, (TP1+FN1))
print('Sen1 of Best_data with random forest is: ', Sen1)
Spec1 = np.divide(TN1, (FP1+TN1))
print('Spec1 of Best_data with random forest is: ', Spec1)
Acc1 = np.divide((TP1+TN1),(TP1+FP1+FN1+TN1))
print('Acc1 of Best_data with random forest is: ',Acc1)

###############################################
# Classify whole data with Adaboost
###############################################
import sklearn
from sklearn.ensemble import  AdaBoostClassifier
c2=1
y2_Best_data = np.array([])
yhat2_Best_data = np.array([])
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clfAdaboost = AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=None))
    clfAdaboost.fit(X_train, y_train)
    #print('Spars Data Accuracy for fold:',c3,clf3.score(X_test, y_test))
    c2 +=1
    y2_Best_data = np.concatenate([y2_Best_data, y_test])
    yhat2 = clfAdaboost.predict(X_test)
    yhat2_Best_data = np.concatenate([yhat2_Best_data, yhat2])
    #metrics.plot_roc_curve(clfAdaboost, X_test, y_test )
    #plt.show()
    score= clfAdaboost.decision_function(X_test)
    fpr,tpr,_ = metrics.roc_curve(y_test,score, pos_label = clfAdaboost.classes_[1])
    metrics.RocCurveDisplay(fpr = fpr, tpr = tpr).plot()
    plt.show()
CM2 = confusion_matrix(y2_Best_data, yhat2_Best_data)
FN2 = CM2[1][0]
TP2 = CM2[1][1]
TN2 = CM2[0][0]
FP2 = CM2[0][1]
Sen2 = np.divide(TP2, (TP2+FN2))
print('Sen2 of Best_data with Adaboost is: ', Sen2)
CM2 = CM2.astype(float)
Spec2 = np.divide(TN2, (FP2+TN2))
print('Spec2 of Best_data with Adaboost is: ', Spec2)
Acc2 = np.divide((TP2+TN2),(TP2+FP2+FN2+TN2))
print('Acc2 of Best_data with Adaboost is: ',Acc2)

###############################################
# Classify whole data with svm
###############################################
from sklearn import svm
c3=1
y3_Best_data = np.array([])
yhat3_Best_data = np.array([])
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clfSVM = svm.LinearSVC()
    clfSVM.fit(X_train, y_train)
    #print('Spars Data Accuracy for fold:',c3,clf3.score(X_test, y_test))
    c3 +=1
    y3_Best_data = np.concatenate([y3_Best_data, y_test])
    yhat3 = clfSVM.predict(X_test)
    yhat3_Best_data = np.concatenate([yhat3_Best_data, yhat3])

CM3 = confusion_matrix(y3_Best_data, yhat3_Best_data)
CM3 = CM3.astype(float)
TN3 = CM3[0][0]
FN3 = CM3[1][0]
TP3 = CM3[1][1]
FP3 = CM3[0][1]
Sen3 = np.divide(TP3, (TP3+FN3))
print('Sen3 of Best_data with svm is: ', Sen3)
Spec3 = np.divide(TN3, (FP3+TN3))
print('Spec3 of Best_data with svm is: ', Spec3)
Acc3 = np.divide((TP3+TN3),(TP3+FP3+FN3+TN3))
print('Acc3 of Best_data with svm is: ',Acc3)
###############################
# Permutation importance
###############################
#from sklearn.inspection import permutation_importance
#result = permutation_importance(clf_Rf, data,labels, n_repeats=1,random_state=42)
###############################
# Feature selection RFE(Recuresive Feature Elimination)
###############################
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

#estimator  = SVR(kernel="linear")
#selector = RFE(estimator, n_features_to_select=20 , step=1)
#selector = selector.fit(data, labels)

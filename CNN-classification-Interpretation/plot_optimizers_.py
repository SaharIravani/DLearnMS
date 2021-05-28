import numpy as np
import matplotlib.pyplot as plt

#Read history of simulated data
N = 10
#fold_no = 1
def plot_optimizer(fold_no):
    history = np.load('/home/sahar/Documents/LC-MS-biomarkerDetection-FinalCode/fold'+str(fold_no)+'_history.npy')
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.item()["loss"], label="train_loss", linestyle= 'solid', linewidth =1.5)
    plt.plot(np.arange(0, N), history.item()["val_loss"], label="val_loss", linestyle= 'dotted', linewidth =1.5)
    plt.plot(np.arange(0, N), history.item()["acc"], label="train_acc", linestyle= 'dashed', linewidth =1.5)
    plt.plot(np.arange(0, N), history.item()["val_acc"], label="val_acc", linestyle= 'dashdot', linewidth =1.5)
    plt.title('fold'+str(fold_no))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower right")
    plt.savefig('plot/'+str(fold_no)+'_plot.png', dp=300, transparent=True)

for fold_no in range(1,6):
    plot_optimizer(fold_no)

#fold_no = 1
N=30
def plot_optimizer_real_data(fold_no):
    history = np.load('/home/sahar/Documents/LC-MS-biomarkerDetection-FinalCode/realData_fold'+str(fold_no)+'_history.npy')
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.item()["loss"], label="train_loss", linestyle= 'solid', linewidth =1.5)
    plt.plot(np.arange(0, N), history.item()["val_loss"], label="val_loss", linestyle= 'dotted', linewidth =1.5)
    plt.plot(np.arange(0, N), history.item()["acc"], label="train_acc", linestyle= 'dashed', linewidth =1.5)
    plt.plot(np.arange(0, N), history.item()["val_acc"], label="val_acc", linestyle= 'dashdot', linewidth =1.5)
    plt.title('fold'+str(fold_no))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower right")
    plt.savefig('plot/realData'+str(fold_no)+'_plot.png', dp=300,transparent =True)

for fold_no in range(1,11):
    plot_optimizer_real_data(fold_no)
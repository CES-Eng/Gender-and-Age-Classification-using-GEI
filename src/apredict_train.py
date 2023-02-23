from datetime import datetime
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

from model.initialization import initialization
from model.utils import evaluation
from configapredict import conf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten



batchsize = 10000
input_columns = 0
rows = 0




def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

def batch_generator(df):
    rows = 0
    idx=0
    dffile = os.path.join(os.path.dirname(os.path.abspath(__file__)),"{}{}.csv").format(df,idx)
    #while (os.path.isfile(dffile)):
    while True:
        values = load_data(dffile)
        yield values[0]
        idx+=1
        rows += values[1]
        dffile = os.path.join(os.path.dirname(os.path.abspath(__file__)),"{}{}.csv").format(df,idx)
        if not os.path.isfile(dffile):
            idx=0
            dffile = os.path.join(os.path.dirname(os.path.abspath(__file__)),"{}{}.csv").format(df,idx)


def load_data(dffile):
    dataset = pd.read_csv(dffile,low_memory=False,dtype=float)
    X = dataset.loc[:,dataset.columns!='Male?']
    Y = dataset.loc[:,dataset.columns=='Male?']
    input_columns = len(dataset.columns)-1
    return [(np.array(X),np.array(Y)),len(dataset.index)]
    
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--iter', default='25000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 1000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()

####################### FCL PART
ffile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'agefeatures(train)0.h5')
if not (os.path.isfile(ffile)):
    m = initialization(conf, train=opt.cache)[0]
    # load model checkpoint of iteration opt.iter
    print('Loading the model of iteration %d...' % opt.iter)
    m.load(opt.iter)
    #extract features
    print('Transforming...')
    test = m.transform('train', opt.batch_size)
    feature, view, seq_type, label = test
    #Create h5 files of certain batch sizes.
    for i in range(0,math.ceil(len(feature)/batchsize)):
        df1 = pd.DataFrame(np.array(feature[batchsize*i:min(len(feature),batchsize*(i+1))]))
        dffeature =df1
        dffeature['ID']= np.array(label[batchsize*i:min(len(label),batchsize*(i+1))]).astype(int)

        #output csv file
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        dfgender = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'OUMVLPinfo.csv'))
        df = pd.merge(dffeature,dfgender)
        del df["ID"]
        dfstring = os.path.join(os.path.dirname(os.path.abspath(__file__)),"agefeatures({}){}.h5")
        df.to_hdf(dfstring.format("train",i), key="ou",mode="w")
        print("fileout")

#split into features and outputs
train_batch = batch_generator("agefeatures(train)")
val_batch = batch_generator("agefeatures(val)")

###################### Count Batches
spe=0
valsteps=0
for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    if "agefeatures(train)"in file:
        spe+=1
    elif "agefeatures(val)"in file:
        valsteps+=1
        
time = datetime.now()
#define fully connected layer
FCLmodel = Sequential([
    Flatten(),
    Dense(32, activation='relu', input_shape=(input_columns,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])
FCLmodel.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'],)


hist = FCLmodel.fit(train_batch,
                    #batch_size=32,
                    epochs=100,
                    validation_data=val_batch,
                    steps_per_epoch=spe,
                    validation_steps=valsteps
                              )

print('Training time:', datetime.now() - time)

FCLmodel.save('/content/drive/MyDrive/ageFCL')

FCLmodel.evaluate(val_batch)[1]

#############classification report and confusion matrix
target_names = ["Female","Male"]
Y_pred = FCLmodel.predict(train_batch,verbose=1)
Y_pred = list(np.around(np.array(Y_pred)))
#Y_pred = np.argmax(Y_pred,axis=1)
print(classification_report(Y_train,Y_pred,target_names=target_names))



cm = confusion_matrix(Y_train,Y_pred)
cm = pd.DataFrame(cm,range(2),range(2))
plt.figure(figsize=(10,10))
ax = sns.heatmap(cm,annot=True, annot_kws={"size":12})
ax.set_title("Confusion Matrix of gender")
ax.set_xlabel("Predicted Gender")
ax.set_ylabel("Actual Gender")
ax.xaxis.set_ticklabels(["Female","Male"])
ax.yaxis.set_ticklabels(["Female","Male"])
plt.show()

##plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()



#plot accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


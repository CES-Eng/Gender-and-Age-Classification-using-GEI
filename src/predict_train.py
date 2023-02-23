from datetime import datetime
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

from model.initialization import initialization
from model.utils import evaluation
from configpredict import conf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()

####################### FCL PART
ffile = 'features(train).csv'
if (os.path.isfile(ffile)):
    df = pd.read_csv(ffile)
    print("file read!")
else:
    m = initialization(conf, train=opt.cache)[0]

    # load model checkpoint of iteration opt.iter
    print('Loading the model of iteration %d...' % opt.iter)
    m.load(opt.iter)

    #extract features
    time = datetime.now()
    print('Transforming...')
    test = m.transform('train', opt.batch_size)
    feature, view, seq_type, label = test


    #assign gender values
    #merge data with values

    print(feature)
    df1 = pd.DataFrame(np.array(feature))
    df2 = pd.DataFrame(np.array(label).astype(int))
    dffeature =df1
    dffeature['ID']= np.array(label).astype(int)

    #output csv file
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    dfgender = pd.read_csv('E:\FYP\Gaitset\casia.csv')
    df = pd.merge(dffeature,dfgender)
    del df["ID"]
    df.to_csv(r'E:\FYP\Gaitset\features(train).csv', encoding='utf-8', index=False)
    print("fileout")

#split into features and outputs
dataset = df.values
X = dataset[:,0:len(df.columns)-1]
print(X.shape)
Y = dataset[:,len(df.columns)-1]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

time = datetime.now()
#define fully connected layer
FCLmodel = Sequential([
    Input(shape=(len(df.columns)-1,)),
    Dense(len(df.columns)/2, activation='relu'),
    Dense(1, activation='sigmoid'),
])
FCLmodel.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = FCLmodel.fit(X_train, Y_train,
          batch_size=32, epochs=200,
          validation_data=(X_val, Y_val))

print('Training time:', datetime.now() - time)

FCLmodel.save('FCL')

FCLmodel.evaluate(X_val, Y_val)[1]

#############classification report and confusion matrix
target_names = ["Female","Male"]
Y_pred = FCLmodel.predict(X_train,verbose=1)
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

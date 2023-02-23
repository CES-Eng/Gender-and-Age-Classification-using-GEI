from datetime import datetime
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

from model.initialization import initialization
from model.utils import evaluation
from config import conf

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()



ffile = 'features(test).csv'
if (os.path.isfile(ffile)):
    df = pd.read_csv(ffile)
    print("file read!")
else:
    m = initialization(conf, test=opt.cache)[0]

    # load model checkpoint of iteration opt.iter
    print('Loading the model of iteration %d...' % opt.iter)
    m.load(opt.iter)
    #extract features
    time = datetime.now()
    print('Transforming...')
    test = m.transform('test', opt.batch_size)
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
    df.to_csv(r'E:\FYP\Gaitset\features(test).csv', encoding='utf-8', index=False)
    print("fileout")
    
#split into features and outputs
dataset = df.values
X = dataset[:,0:len(df.columns)-1]
Y = dataset[:,len(df.columns)-1]

#Load model of fully connected layer
print('Loading fully connected layer')
FCLmodel = load_model('ageFCL')

#predict from features
print('Evaluating...')
FCLmodel.evaluate(X, Y)[1]

#print('Evaluation complete. Cost:', datetime.now() - time)

#############classification report and confusion matrix
target_names = ["Female","Male"]
Y_pred = FCLmodel.predict(X,verbose=1)
Y_pred = list(np.around(np.array(Y_pred)))
#Y_pred = np.argmax(Y_pred,axis=1)

cm = confusion_matrix(Y,Y_pred)
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

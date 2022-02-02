import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
import sys
import os.path

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.utils import class_weight
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay,  confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score


import tensorflow as tf
from keras import backend as k
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, Layer, InputSpec
from keras.models import Sequential,Model
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.utils import np_utils, conv_utils

import proplot as pplt
import cartopy.feature as cfeature
import xskillscore as xs

from mlprecip_utils import *
from mlprecip_xai import *

# Turn off deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    
def logmodel_med(in_shape):

    """
    Logistic Regression with same setup as NN Input(nfeatures)->Output(2)

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training in one hot encoding format (nsamples,ncategories)

    Returns:
    model

    """
    def log_model():
        
        model = Sequential()
        
        model.add(Dense(2,activation='softmax',input_dim=in_shape))

        model.compile(optimizer=optimizers.Adam(lr=1e-5),
                      loss='categorical_crossentropy',
                      metrics = ['accuracy'])

   
        return model
    return log_model

def nnmodel_med(in_shape):

    """
    Fit fully connected neural network Input(nfeatures)->8->8->Output(3)

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training in one hot encoding format (nsamples,ncategories)

    Returns:
    model

    """
    def cat_model_med():
        
        model = Sequential()
        
        model.add(Dense(4, input_dim=in_shape,activation='relu',
                        kernel_initializer='he_normal',
                        bias_initializer='he_normal'))
       
        model.add(Dense(2,activation='softmax'))

        model.compile(optimizer=optimizers.Adam(lr=1e-5),
                      loss='categorical_crossentropy',
                      metrics = ['accuracy'])
        

   
        return model
    return cat_model_med

def cnn_cat(input_shape):

    """
    This function was adapted from the happyModel example 
    taken from a deeplearning.ai course taught by Andrew Ng on Coursera. 
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    Returns:
    model -- a Model() instance in Keras
    """
    def cnn_model():
        
        X_input=Input(shape=input_shape)
        
        # Layer 1: CONV1->RELU->MAXPOOL
        X=Conv2D(filters=16, kernel_size=(3,3), 
                 strides = (1,1), padding='valid',
                 kernel_regularizer=regularizers.l2(20),
                 activation='relu',name='c1')(X_input)
        X=MaxPooling2D((3,3),strides=(1,1),padding='valid',name='mp1')(X)
        
        # Layer 2: CONV2->RELU>MAXPOOL
        X=Conv2D(filters=32, kernel_size=(3,3), 
                 strides = (1, 1),
                 padding='valid',activation='relu',
                kernel_regularizer=regularizers.l2(10))(X)
        X=MaxPooling2D((3,3),strides=(1,1),padding='valid')(X)

        # Layer 3: CONV3->RELU>MAXPOOL
        X=Conv2D(filters=64, kernel_size=(3,3), 
                 strides = (1, 1),
                 padding='valid',activation='relu')(X)
        X=MaxPooling2D((3,3),strides=(1,1),padding='valid')(X)
        
        X=Flatten()(X)
             
        # Layer 4: FC1->RELU
        X=Dense(128, activation='relu',name='fc1')(X)
        
        # Output Layer
        X=Dense(2, activation='softmax',name='dense_output')(X)
    
        model = Model(inputs = X_input, outputs = X)
        
        model.compile(optimizer=optimizers.Adam(lr=3e-5),
                      loss='categorical_crossentropy',
                      metrics = ['accuracy'])

        return model
    
    return cnn_model

def trainIndexModels(model_func,ds_features,ds_target,v,nmodels,fname='',ofname=''):
    
    feature_vars=list(ds_features.keys())
    # Setup Features (X) and Target (Y)
    
    X=ds_features.to_stacked_array('features',sample_dims=['time']).values
    Y=make_ohe_thresh_med(ds_target[v])
    cat_labels=['Negative','Positive']

    print('Check Features and Target Dimensions')
    print('Features (X): ',X.shape)
    print('Target (Y): ',Y.shape)
    
    # ---------- Create Train and Validation Sets ---------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.9,shuffle=False)

    ntrain=X_train.shape[0]
    ntest=X_test.shape[0]

    print('Training Size: ',ntrain)
    print('Validation Size: ',ntest)    
    
    for i in range(nmodels):
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        call_model=getattr(sys.modules[__name__],model_func)(X_train.shape[1])
        nn = KerasClassifier(build_fn=call_model,epochs=250,batch_size=25, verbose=0)
        #history=nn.fit(X_train, Y_train,validation_data=(X_test,Y_test),callbacks=[es])
        history=nn.fit(X_train, Y_train,validation_data=(X_test,Y_test))
        
        plotLearningCurve(history)
        
        # Predict category and probs
        Ypred_test=nn.predict(X_test)
        Yprobs_test=nn.predict_proba(X_test)
        
        Ypred_train=nn.predict(X_train)
        Yprobs_train=nn.predict_proba(X_train)
        
        # Save Model
        if (fname):
            nn.model.save(fname+'.'+str(i)+'.h5')
                    
        # Scores and Check
        print('Training set accuracy score: ' + str(nn.score(X_train, Y_train)))
        print('Validation set accuracy score: ' + str(nn.score(X_test, Y_test)))
        print('Validation ROC AUC score: ' + str(roc_auc_score(Y_test, nn.predict_proba(X_test), multi_class='ovr')))
        
        # Calculate LRP
        rules=['lrp.alpha_1_beta_0','lrp.z']
        a=calcLRP(nn.model,X.reshape(X.shape[0],X.shape[1]),rules=rules)
        b=np.asarray(a)
                
        # Put all model output information into a Dataset to be written to a netcdf file 
        ds_lrp=xr.DataArray(b,
                            coords={'rules':rules,
                                    'time':ds_features['time'],
                                    'var':feature_vars},
                            dims=['rules','time','var']).to_dataset(name='lrp')    
        
        ds_pred=xr.DataArray(np.concatenate([Ypred_train,Ypred_test]),
                             coords={'time':ds_features['time']},
                             dims=['time']).to_dataset(name='pred')
    
        ds_probs=xr.DataArray(np.concatenate([Yprobs_train,Yprobs_test]),
                              coords={'time':ds_features['time'],
                                      'cat':cat_labels},
                              dims=['time','cat']).to_dataset(name='probs')
    
        ds_acc=xr.DataArray(nn.score(X_train, Y_train),
                            coords={'model':[i]},
                            dims=['model']).to_dataset(name='acc')
    
        ds_valacc=xr.DataArray(nn.score(X_test, Y_test),
                               coords={'model':[i]},
                               dims=['model']).to_dataset(name='val_acc')
    
        ds_verif=xr.DataArray(np.concatenate([np.argmax(Y_train,axis=1),np.argmax(Y_test,axis=1)]),
                              coords={'time':ds_features['time']},
                              dims=['time']).to_dataset(name='verif')
          
        ds=xr.merge([ds_lrp,ds_pred,ds_verif,ds_probs,ds_acc,ds_valacc,ds_target])
    
        # Write all model output information 
        if (ofname):
            model_output_fname=ofname+'.'+str(i)+'.nc'
            ds.to_netcdf(model_output_fname) 
        
    
    return 


def trainCNN(model_func,ds_features,ds_target,varname,nmodels,fname='',ofname=''):

    # --------  Setup Features (X) and Target (Y) ----------------------------------
    pad_length=10
    feature_vars=list(ds_features.keys())
    da_list=[]
    for v in feature_vars:
        da_list.append(ds_features[v])
        X=xr.combine_nested(da_list,concat_dim='var') 
        X=(X.transpose('time','lat','lon','var')).values
        X=xr.where(X!=0,(X-np.nanmean(X,axis=0))/np.nanstd(X,axis=0),0.0)
        X_pad=np.pad(X,((0,0),(0,0),(pad_length,pad_length),(0,0)),'wrap')
    
    # One Hot Encode Target (Y)
    Y=make_ohe_thresh_med(ds_target[varname])
    cat_labels=['Negative','Positive']

    print('Check Features and Target Dimensions')
    print('Features (X): ',X_pad.shape)
    print('Target (Y): ',Y.shape)
    
    # ---------- Create Train and Validation Sets ---------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(X_pad,Y,train_size=0.9,shuffle=False)

    ntrain=X_train.shape[0]
    ntest=X_test.shape[0]

    print('Training Size: ',ntrain)
    print('Validation Size: ',ntest)    
    
    #---------- Loop over all models to train and validate ------------------------
    
    for i in range(83,nmodels):
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
        
        if len(X_train.shape[1:])==1:
            in_shape=X_train.shape[1]
        else:
            in_shape=X_train.shape[1:]
            
        call_model=getattr(sys.modules[__name__],model_func)(in_shape)
        
        nn = KerasClassifier(build_fn=call_model,epochs=100,batch_size=25, verbose=0)
        history=nn.fit(X_train, Y_train,validation_data=(X_test,Y_test),callbacks=[es])

        plotLearningCurve(history)
        
        # Predict category and probs
        Ypred_test=nn.predict(X_test)
        Yprobs_test=nn.predict_proba(X_test)
        
        Ypred_train=nn.predict(X_train)
        Yprobs_train=nn.predict_proba(X_train)
        
        # Save Model
        if (fname):
            nn.model.save(fname+'.'+str(i)+'.h5')
                    
        # Scores and Check
        print('Training set accuracy score: ' + str(nn.score(X_train, Y_train)))
        print('Validation set accuracy score: ' + str(nn.score(X_test, Y_test)))
        print('Validation ROC AUC score: ' + str(roc_auc_score(Y_test, nn.predict_proba(X_test), multi_class='ovr')))
        
        # Calculate LRP (TO-DO: put this into and return ds_lrp)
        rules=['lrp.alpha_1_beta_0','lrp.z']
        a=calcLRP(nn.model,X_pad.reshape(X_pad.shape[0],
                                         X_pad.shape[1],
                                         X_pad.shape[2],
                                         X_pad.shape[3]),rules=rules)
        b=np.asarray(a)[:,:,:,pad_length:-pad_length,:]
        
        # Put all model output information into a Dataset to be written to a netcdf file 
        ds_lrp=xr.DataArray(b,
                            coords={'rules':rules,
                                    'time':ds_features['time'],
                                    'lat':ds_features['lat'],
                                    'lon':ds_features['lon'],
                                    'var':feature_vars},
                            dims=['rules','time','lat','lon','var']).to_dataset(name='lrp')    
        
        ds_pred=xr.DataArray(np.concatenate([Ypred_train,Ypred_test]),
                             coords={'time':ds_features['time']},
                             dims=['time']).to_dataset(name='pred')
    
        ds_probs=xr.DataArray(np.concatenate([Yprobs_train,Yprobs_test]),
                              coords={'time':ds_features['time'],
                                      'cat':cat_labels},
                              dims=['time','cat']).to_dataset(name='probs')
    
        ds_acc=xr.DataArray(nn.score(X_train, Y_train),
                            coords={'model':[i]},
                            dims=['model']).to_dataset(name='acc')
    
        ds_valacc=xr.DataArray(nn.score(X_test, Y_test),
                               coords={'model':[i]},
                               dims=['model']).to_dataset(name='val_acc')
    
        ds_verif=xr.DataArray(np.concatenate([np.argmax(Y_train,axis=1),np.argmax(Y_test,axis=1)]),
                              coords={'time':ds_features['time']},
                              dims=['time']).to_dataset(name='verif')
          
        ds=xr.merge([ds_lrp,ds_pred,ds_verif,ds_probs,ds_acc,ds_valacc,ds_target])
    
        # Write all model output information 
        if (ofname):
            model_output_fname=ofname+'.'+str(i)+'.nc'
            ds.to_netcdf(model_output_fname) 
        
    return 
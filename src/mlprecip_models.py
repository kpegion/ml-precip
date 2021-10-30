import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
import sys

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.utils import class_weight
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay,  confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score


import tensorflow as tf
from keras import backend as k
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.utils import np_utils

import proplot as pplt
import cartopy.feature as cfeature
import xskillscore as xs

from mlprecip_utils import *
from mlprecip_xai import *

# Turn off deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def lasso(X,Y):

    """
    Fit regression model using Lasso (R1) regularization

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model, coefficients, r-squared, and the predicted values of Y

    """

    regr = LassoCV(cv=5,max_iter=5000).fit(X,Y)
    pred = regr.predict(X)

    return regr,pred

def ridge(X,Y):

    """
    Fit regression model using Ridge (R2) regularization

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model, coefficients, r-squared, and the predicted values of Y

    """

    regr = RidgeCV(cv=5).fit(X,Y)
    pred = regr.predict(X)

    return regr,pred

def lr(X,Y):

    """
    Fit regression model using standard regression model without regularization

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model, coefficients, r-squared, and the predicted values of Y

    """

    regr = LinearRegression().fit(X,Y)
    pred = regr.predict(X)

    return regr,pred

def logistic(X,Y):

    """
    Fit regression model using standard logistic regression model. Uses balanced class weighting in 
    case target classes are imbalanced.

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model, coefficients, r-squared, and the predicted values of Y

    """

    regr = LogisticRegression(penalty='none',multi_class='multinomial',solver='sag').fit(X,Y)
    pred = regr.predict(X)
    
    cm = confusion_matrix(Y, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    print(classification_report(Y, pred))


    return regr,pred

def tomsensomodel_regression(in_shape):

    """
    Fit fully connected neural network Input(nfeatures)->8->8->Output(1)

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model

    """
    
    def regr_model():
        model = Sequential()

        model.add(Dense(8, input_dim=in_shape,activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.01),
                bias_initializer='he_normal'))

        model.add(Dense(8, activation='tanh',
                    kernel_initializer='he_normal',
                    bias_initializer='he_normal'))
        
        model.add(Dense(1,name='output'))

        model.compile(optimizer=optimizers.Adam(),
                      loss ='mean_squared_error',
                      metrics = [r2_score])
    
        return model
    return regr_model

def tomsensomodel_cat_terc(in_shape):

    """
    Fit fully connected neural network Input(nfeatures)->8->8->Output(3)

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training in one hot encoding format (nsamples,ncategories)

    Returns:
    model

    """
    def cat_model_terc():
        
        model = Sequential()
        model.add(Dense(8, input_dim=in_shape,activation='relu',
                        kernel_initializer='he_normal',
                        bias_initializer='he_normal'))
        
        model.add(Dense(8, activation='relu',
                kernel_initializer='he_normal',
                bias_initializer='he_normal'))
        
        model.add(Dense(3,activation='softmax'))

        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                      loss='categorical_crossentropy',
                      metrics = ['accuracy'])
   
        return model
    return cat_model_terc

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

        model.compile(optimizer=optimizers.Adam(lr=0.0001),
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
       
        #model.add(Dense(8, activation='relu',
        #        kernel_initializer='he_normal',
        #        bias_initializer='he_normal'))
        
        model.add(Dense(2,activation='softmax'))

        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                      loss='categorical_crossentropy',
                      metrics = ['accuracy'])
        

   
        return model
    return cat_model_med

def cnn_cat(input_shape):

    """
    Implementation of the Ham et al. CNN ENSO Model prior to transfer learning.
    This function was adapted from the happyModel example 
    taken from a deeplearning.ai course taught by Andrew Ng on Coursera. 
    It was adapted to match the CNN model of ENSO used in Ham et al. 
    
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
        
        # Define the input placeholder as a tensor with shape input_shape. 
        X_input = Input(input_shape)

        # Layer 1: CONV1->TANH->MAXPOOL
        X = Conv2D(filters=2, kernel_size=(4,2), 
                   strides = (1, 1), 
                   padding='same', 
                   name = 'conv1',kernel_regularizer=regularizers.l1(25))(X_input)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2),strides=(2,2),
                         name='max_pool1',padding='valid')(X)
        X = Dropout(0.2)(X)
        X = Activation('relu')(X)
        X = Flatten()(X)
    
        # Layer 4: FC1->TANH
        X = Dense(8, activation='relu', name='fc1',kernel_regularizer=regularizers.l2(0.02))(X)
        
        # Layer 4: FC1->TANH
        X = Dense(4, activation='relu', name='fc2')(X)

        # Output Layer
        X = Dense(1, activation='sigmoid',name='output')(X)
    
        # Create model
        model = Model(inputs = X_input, outputs = X, name='cnn')

        # Compile Model
        model.compile(optimizer=optimizers.Adam(lr=0.0005),
                      loss='binary_crossentropy',
                      metrics = ['accuracy'])
        return model
    
    return cnn_model

def testModelsCat(model_func,ds_features,ds_target,v,nmodels,fname=''):
    
    # Setup Features (X) and Target (Y)
    
    X=ds_features.to_stacked_array('features',sample_dims=['time'])
    Y=make_ohe_thresh_med(ds_target[v])
    cat_labels=['Lower','Upper']

    print('Check Features and Target Dimensions')
    print('Features (X): ',X.shape)
    print('Target (Y): ',Y.shape)

    nsamples=X.shape[0]
    nfeatures=X.shape[1]

    print("Samples: ",nsamples)
    print("Features: ", nfeatures)
    
    # Create Train and Test Sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,shuffle=False)

    ntrain=X_train.shape[0]
    ntest=X_test.shape[0]

    print('Training Size: ',ntrain)
    print('Testing Size: ',ntest)    
    
    acc_list=[]
    valacc_list=[]
    pred_list=[]
    probs_list=[]
    lrp_list=[]
    verif_list=[]
    
    for i in range(nmodels):
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        call_model=getattr(sys.modules[__name__],model_func)(X_train.shape[1])
        nn = KerasClassifier(build_fn=call_model,epochs=250,batch_size=25, verbose=0)
        history=nn.fit(X_train, Y_train,validation_data=(X_test,Y_test),callbacks=[es])
        
        
        # Predict category and probs
        Ypred_test=nn.predict(X_test)
        Yprobs_test=nn.predict_proba(X_test)
        
        Ypred_train=nn.predict(X_train)
        Yprobs_train=nn.predict_proba(X_train)
        
        # Save model, history, and predictions
        pred_list.append(np.concatenate([Ypred_train,Ypred_test]))
        probs_list.append(np.concatenate([Yprobs_train,Yprobs_test]))
        verif_list.append(np.concatenate([np.argmax(Y_train,axis=1),np.argmax(Y_test,axis=1)]))
        
        if (fname):
            nn.model.save(fname+'.'+str(i)+'.h5')
        
        # Classification Report
        #print(classification_report(np.argmax(Y_test,axis=1), Ypred_nn))

        # Scores and Check
        print('Training set accuracy score: ' + str(nn.score(X_train, Y_train)))
        print('Test set accuracy score: ' + str(nn.score(X_test, Y_test)))
        print('Test ROC AUC score: ' + str(roc_auc_score(Y_test, nn.predict_proba(X_test), multi_class='ovr')))
    
        acc_list.append(nn.score(X_train, Y_train))
        valacc_list.append(nn.score(X_test, Y_test))
    
        # Calculate LRP 
        rules=['lrp.alpha_1_beta_0','lrp.z']
            
        a=calcLRP(nn.model,X,rules=rules)
        ds_tmp=xr.DataArray(np.asarray(a).reshape(len(rules),nsamples,nfeatures),
                            coords={'rules':rules,'time':ds_features['time'],
                            'features':list(ds_features.keys())},
                            dims=['rules','time','features'])        
        ds_tmp=ds_tmp.to_dataset(name='lrp')
                
        # Save LRP for this model
        lrp_list.append(ds_tmp)
        
    ds_lrp=xr.combine_nested(lrp_list,concat_dim='model')
    ds_lrp['model']=np.arange(nmodels)
    
    ds_pred=xr.DataArray(np.asarray(pred_list).squeeze(),
                         coords={'model':np.arange(nmodels),
                                 'time':ds_features['time']},
                         dims=['model','time']).to_dataset(name='pred')
    
    ds_probs=xr.DataArray(np.asarray(probs_list).squeeze(),
                          coords={'model':np.arange(nmodels),
                                  'time':ds_features['time'],
                                  'cat':cat_labels},
                          dims=['model','time','cat']).to_dataset(name='probs')
    
    ds_acc=xr.DataArray(np.asarray(acc_list).squeeze(),
                        coords={'model':np.arange(nmodels)},
                        dims=['model']).to_dataset(name='acc')
    
    ds_valacc=xr.DataArray(np.asarray(valacc_list).squeeze(),
                           coords={'model':np.arange(nmodels)},
                           dims=['model']).to_dataset(name='val_acc')
    
    ds_verif=xr.DataArray(np.asarray(verif_list).squeeze(),
                          coords={'model':np.arange(nmodels),
                                  'time':ds_features['time']},
                          dims=['model','time']).to_dataset(name='verif')
  
    ds=xr.merge([ds_lrp,ds_pred,ds_verif,ds_probs,ds_acc,ds_valacc,ds_target])

    
    return ds


def testModelsRegr(ds_features,ds_target):

    # Setup Features (X) and Target (Y)
    
    X=ds_features.to_stacked_array('features',sample_dims=['time'])
    Y=ds_target['precip'].values

    #print('Check Features and Target Dimensions')
    #print('Features (X): ',X.shape)
    #print('Target (Y): ',Y.shape)

    nsamples=X.shape[0]
    nfeatures=X.shape[1]

    #print("Samples: ",nsamples)
    #print("Features: ", nfeatures)
    
    # Create Train and Test Sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,shuffle=False)

    ntrain=X_train.shape[0]
    ntest=X_test.shape[0]

    #print('Training Size: ',ntrain)
    #print('Testing Size: ',ntest)
    
    # Make a heatmap
    heatmap(X_train,Y_train,list(ds_features.keys()))
    
    # Train the Models
    
    # -- Standard Linear Regression
    regr_lr,Ypred_lr=lr(X_train,Y_train)
    print('Regression Training set R^2 score: ' + str(regr_lr.score(X_train,Y_train)))
    print('Regression Test set R^2 score: ' + str(regr_lr.score(X_test,Y_test)))

    # -- Linear Regression with LASSO Regularization
    regr_lasso,Ypred_lasso=lasso(X_train,Y_train)
    print('LASSO Training set R^2 score: ' + str(regr_lasso.score(X_train,Y_train)))
    print('LASSO Test set R^2 score: ' + str(regr_lasso.score(X_test,Y_test)))

    # -- Linear Regression with Ridge Regularization
    regr_ridge,Ypred_ridge=ridge(X_train,Y_train)
    print('Ridge Training set R^2 score: ' + str(regr_ridge.score(X_train,Y_train)))
    print('Ridge Test set R^2 score: ' + str(regr_ridge.score(X_test,Y_test)))

    # -- Neural Network
    nn = KerasRegressor(build_fn=tomsensomodel_regression(X_train.shape[1]),epochs=250, batch_size=25,verbose=0)
    history=nn.fit(X_train, Y_train)
    Ypred_nn=nn.predict(X_train)
    print('NN Training set R^2 score: ' + str(nn.score(X_train, Y_train)))
    print('NN Test set R^2 score: ' + str(nn.score(X_test, Y_test)))

    # Plot target and fit 
    plt.figure(figsize=(11,8.5))
    plt.plot(Y_train)
    plt.plot(Ypred_lr)
    plt.plot(Ypred_ridge)
    plt.plot(Ypred_lasso)
    plt.plot(Ypred_nn)
    plt.legend(['Target','Standard','Ridge','LASSO','NN'])
    
    # Plot Coefficients for Standard Linear Regression
    plt.figure(figsize=(11,8.5))
    plt.bar(list(ds_features.keys()),regr_lr.coef_)              

    return

def testModelsCatField(ds_features,ds_target,nmodels):
    
    nx=len(ds_features['lon'])
    ny=len(ds_features['lat'])
    nvar=2
    # Setup Features (X) and Target (Y)
    
    X=xr.combine_nested([ds_features['z500'],ds_features['u250']],concat_dim='var')
    X=X.transpose('time','lat','lon','var')
    Y=ds_target['precip'].values

    print('Check Features and Target Dimensions')
    print('Features (X): ',X.shape)
    print('Target (Y): ',Y.shape)

    nsamples=X.shape[0]
    nfeatures=X.shape[1:]

    print("Samples: ",nsamples)
    print("Features: ", nfeatures)
    
    # Create Train and Test Sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,shuffle=False)

    ntrain=X_train.shape[0]
    ntest=X_test.shape[0]

    print('Training Size: ',ntrain)
    print('Testing Size: ',ntest)
    
    # Train the Models
    acc_list=[]
    valacc_list=[]
    pred_list=[]
    probs_list=[]
    lrp_list=[]
    verif_list=[]
    
    for i in range(nmodels):
        
        # -- Neural Network -- #
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
        weights = class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train)
        nn = KerasClassifier(build_fn=cnn_cat(X_train.shape[1:]),epochs=50, batch_size=256,verbose=0,class_weight=weights)
        history=nn.fit(X, Y,validation_split=0.2,shuffle=False,callbacks=[es])
        Ypred_nn=nn.predict(X_train)
        Yprobs_nn=nn.predict_proba(X_train)
        
        cm = confusion_matrix(Y_train, Ypred_nn)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['In Cat','Out Cat'])
        disp.plot()
        
        # Save model, history, and predictions
        pred_list.append(Ypred_nn)
        probs_list.append(Yprobs_nn)
        verif_list.append(Y_train)
        nn.model.save('../data/cnn/upper_tercile_seus.'+str(i)+'.h5')

        # Print checks for this model
        print("CHECK NN: ","In Category ", np.count_nonzero(Ypred_nn==0)," Not in category: ",np.count_nonzero(Ypred_nn==1))
        print('NN Training set accuracy score: ' + str(nn.score(X_train, Y_train)))
        print('NN Test set accuracy score: ' + str(nn.score(X_test, Y_test)))
        
        acc_list.append(nn.score(X_train, Y_train))
        valacc_list.append(nn.score(X_test, Y_test))
    
        # Calculate LRP 
        a=calcLRP(nn.model,X_train)
        ds_tmp=xr.DataArray(a.reshape(ntrain,ny,nx,nvar),
                        coords={'time':ds_features['time'][0:ntrain],
                                'lat':ds_features['lat'],
                                'lon':ds_features['lon'],
                                'var':['z500','u250']},
                        dims=['time','lat','lon','var'])        
        ds_tmp=ds_tmp.to_dataset(name='lrp')
    
        # Normalize by max value in grid
        ds_tmp=ds_tmp/ds_tmp.max(dim=['lat','lon'])
        
        # Save LRP for this model
        lrp_list.append(ds_tmp)
    
    ds_lrp=xr.combine_nested(lrp_list,concat_dim='model')
    ds_lrp['model']=np.arange(nmodels)
    
    ds_pred=xr.DataArray(np.asarray(pred_list).squeeze(),
                         coords={'model':np.arange(nmodels),
                                 'time':ds_features['time'][0:ntrain]},
                         dims=['model','time']).to_dataset(name='pred')
    
    ds_probs=xr.DataArray(np.asarray(probs_list).squeeze(),
                          coords={'model':np.arange(nmodels),
                                  'time':ds_features['time'][0:ntrain],
                                  'cat':['True','False']},
                          dims=['model','time','cat']).to_dataset(name='probs')
    
    ds_acc=xr.DataArray(np.asarray(acc_list).squeeze(),
                        coords={'model':np.arange(nmodels)},
                        dims=['model']).to_dataset(name='acc')
    
    ds_valacc=xr.DataArray(np.asarray(valacc_list).squeeze(),
                           coords={'model':np.arange(nmodels)},
                           dims=['model']).to_dataset(name='val_acc')
    
    ds_verif=xr.DataArray(np.asarray(verif_list).squeeze(),
                          coords={'model':np.arange(nmodels),
                                  'time':ds_features['time'][0:ntrain]},
                          dims=['model','time']).to_dataset(name='verif')
  
    ds=xr.merge([ds_lrp,ds_pred,ds_verif,ds_probs,ds_acc,ds_valacc])
    
    
    return ds
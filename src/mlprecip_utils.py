import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from sklearn.utils import class_weight

import tensorflow as tf
from keras import backend as k
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import proplot as pplt
import cartopy.feature as cfeature

import innvestigate
import innvestigate.utils

# Turn off deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def r2_score(y_true, y_pred):
    SS_res =  k.sum(k.square( y_true-y_pred ))
    SS_tot = k.sum(k.square( y_true - k.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + k.epsilon()) )


def init_predictors_dict():
    
    amo_dict=dict(name='amo',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                 file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/amo.txt')
    naomonthly_dict=dict(name='nao',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                        file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/nao.txt')
    nino34_dict=dict(name='nino34',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                    file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/nino34.txt')
    pdo_dict=dict(name='pdo',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                 file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/pdo.txt')
    rmmamp_dict=dict(name='RMM_amp',ptype='index',freq='day',readfunc='getRMM',
                     file='/data/ccsm4/kpegion/obs2/RMM/rmmint1979-092021.txt')
    rmmphase_dict=dict(name='RMM_phase',ptype='cat',freq='day',readfunc='getRMM',
                      file='/data/ccsm4/kpegion/obs2/RMM/rmmint1979-092021.txt')
    pnaregimes_dict=dict(name='pnaregimes',ptype='cat',freq='day',readfunc='getWR',
                         file='/scratch/kpegion/ERAI_clusters_5_1980-2015_')
    mlso_dict=dict(name='mlso',ptype='index',freq='day',readfunc='getMLSO',
                  file='/data/vortex/scratch/mlso.index.01011979-08312019.nc')
    nashamp_dict=dict(name='nash_amp',ptype='index',freq='day',readfunc='getNASH',
                      file='/scratch/kpegion/ERAI_NASH_JJA.1997-2015.nc')
    nashphase_dict=dict(name='nash_phase',ptype='cat',freq='day',readfunc='getNASH',
                        file='/scratch/kpegion/ERAI_NASH_JJA.1997-2015.nc')

    predictors=[amo_dict,naomonthly_dict,nino34_dict,pdo_dict,rmmamp_dict,rmmphase_dict,mlso_dict,pnaregimes_dict,nashphase_dict,nashamp_dict]
    
    return predictors

def init_fields_dict():
    
    pna_z500_dict=dict(name='pnaz500',freq='day',readfunc='getfield',
                       file='/scratch/kpegion/ERAI_clusters_5_1980-2015_')

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

    regr = LogisticRegression(class_weight='balanced').fit(X,Y)
    pred = regr.predict(X)

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
                kernel_regularizer=regularizers.l1(0.02),
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

def tomsensomodel_cat(in_shape):

    """
    Fit fully connected neural network Input(nfeatures)->8->8->Output(1)

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model

    """
    def cat_model():
        
        model = Sequential()
        model.add(Dense(8, input_dim=in_shape,activation='relu',
                    kernel_initializer='he_normal',
                    bias_initializer='he_normal'))

        model.add(Dense(8, activation='relu',
                    kernel_initializer='he_normal',
                    bias_initializer='he_normal'))
        
        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer=optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics = ['accuracy'])
    
        return model
    return cat_model

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
                   name = 'conv1')(X_input)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2),strides=(2,2),
                         name='max_pool1',padding='valid')(X)
    
        # Layer 2: CONV2->TANH->MAXPOOL
        #X = Conv2D(filters=50, kernel_size=(4,2), 
        #           strides = (1, 1), 
        #           padding='same',
        #           name = 'conv2')(X)
        #X = Activation('relu')(X)
        #X = MaxPooling2D((2, 2),strides=(2,2),
        #                 padding='valid',name='max_pool2')(X)
    
        # Layer 3: CONV3->TANH (then FLATTEN)
        #X = Conv2D(filters=50, kernel_size=(4,2), 
        #           strides = (1, 1), 
        #           padding='same', 
        #           name = 'conv3')(X)
        X = Activation('relu')(X)
        X = Flatten()(X)
    
        # Layer 4: FC1->TANH
        X = Dense(8, activation='relu', name='fc1')(X)

        # Output Layer
        #X = Dense(1,name='output')(X)
        X = Dense(1, activation='sigmoid',name='output')(X)
    
        # Create model
        model = Model(inputs = X_input, outputs = X, name='cnn')

        # Compile Model
        model.compile(optimizer=optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics = ['accuracy'])
        return model
    
    return cnn_model

def heatmap(X,Y,labels):

    """
    Plot the seaborn heatmap of correlations between all values of X and Y

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    Nothing; display plot to screen

    """
    tmp=np.hstack((X,np.expand_dims(Y, axis=1)))
    d = pd.DataFrame(data=tmp,columns=labels.append("target"))
    corr=d.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(11,11))
    sns.set(font_scale=1)
    ax=sns.heatmap(corr,square=True,linewidths=0.5,fmt=" .2f", \
                   annot=True,mask=mask,cmap='seismic', \
                   vmin=-1,vmax=1,cbar=False,annot_kws={"size": 10})
    
def makeCategories(ds,bins,index_name):
    
    nbins=len(bins)-1
   
   # Create bins and assign integer to each bin
    tmp=np.zeros((ds['time'].shape[0]))
    
    for i in range(nbins):
        tmp[(ds[index_name]>=bins[i]) & (ds[index_name]<bins[i+1])] = i
        
    # Put into xarray.Dataset
    ds_tmp=xr.DataArray(tmp,
                        coords={'time':ds['time'].values},
                                dims=['time'])        
    ds_tmp=ds_tmp.to_dataset(name=index_name+'_bins')
    
    return ds_tmp
    
def calcComposites(ds,index_name,labels):
    
    totals=[]
    
    # Determine number of bins
    nbins=len(ds[index_name+'_bins'])
    
    # Loop over bins to calculate number in each bin and print
    for j,l in zip(range(nbins),labels):

        total=ds[index_name+'_bins'].where(ds[index_name+'_bins']==j).count(dim='time')
        totals.append(total)

    # Calculate the composites
    ds_comp_anoms=ds.groupby(index_name+'_bins').mean(dim='time')

    return ds_comp_anoms,totals

def plotComposites(ds,index_name,totals,suptitle,labels,clevs,cmap,figfile):
    
    dim_str=index_name+'_bins'
    nbins=int(np.max(ds[dim_str].values)+1)
    
    # Define map region and center longitude
    lonreg=(269,283)
    latreg=(24,36)
    lon_0=290

    # Set number of rows and columns and define subplots
    if (nbins > 4):
        ncols=2
        nrows=int(np.ceil(nbins/ncols))
    else:
        ncols=1
        nrows=nbins
    
    f, axs = pplt.subplots(ncols=ncols, nrows=nrows,
                           proj='pcarree',proj_kw={'lon_0': lon_0},
                           width=8.5,height=11.0)
    
    
    # Plot all bins
    for i in range(nbins):
        ds_us=ds['precip'].sel({dim_str:i})
        m=axs[i].contourf(ds_us['lon'], ds_us['lat'],
                          ds_us,levels=clevs,
                          cmap=cmap,extend='both')
        axs[i].format(coast=True,lonlim=lonreg,latlim=latreg,grid=True,
                      lonlabels='b', latlabels='l',title=labels[i]+' ('+str(int(totals[i]))+')',
                      suptitle=suptitle,abc=True)
        # Add US state borders    
        axs[i].add_feature(cfeature.STATES,edgecolor='gray')

    # Colorbar
    f.colorbar(m,loc='b',length=0.75)
    
    # Save to file
    plt.savefig(figfile)
    
def getPrecipData(fnames,sdate,edate):
    
    # Read Data
    ds=xr.open_mfdataset(fnames,combine='by_coords').sel(time=slice(sdate,edate))
    
    # Remove duplicate times?
    ds=ds.sel(time=~ds.get_index("time").duplicated())

    # Rename coordinate and drop unused ones
    ds=ds.rename({'latitude':'lat','longitude':'lon'}).reset_coords(['longitude_bnds','latitude_bnds'],drop=True)
    
    return ds

def calcRatios(ds,index_name,v,thresh):
    
    above=(ds[v].where(ds[v]>thresh)).groupby(ds[index_name+'_bins']).count(dim='time')
    below=(ds[v].where(ds[v]<thresh)).groupby(ds[index_name+'_bins']).count(dim='time')
    ratio=above/below 
    
    return above,below,ratio

def plotRatios(da,index_name,suptitle,labels,clevs,cmap,figfile):

    dim_str=index_name+'_bins'
    nbins=int(np.max(da[dim_str].values)+1)
    
    
    # Define map region and center longitude
    lonreg=(269,283)
    latreg=(24,36)
    lon_0=290

    # Set number of rows and columns and define subplots
    if (nbins > 4):
        ncols=2
        nrows=int(np.ceil(nbins/ncols))
    else:
        ncols=1
        nrows=nbins
        
    f, axs = pplt.subplots(ncols=ncols, nrows=nrows,
                           proj='pcarree',proj_kw={'lon_0': lon_0},
                           width=8.5,height=11.0)
    
    #norm = pplt.Norm('diverging', vcenter=1)
    
    # Plot all bins
    for i in range(nbins):
        
        m=axs[i].contourf(da['lon'], da['lat'],
                          da.sel({dim_str:i}),
                          cmap=cmap,extend='both')
        axs[i].format(coast=True,lonlim=lonreg,latlim=latreg,grid=True,
                      lonlabels='b', latlabels='l',title=labels[i],
                      suptitle=suptitle,abc=True)

    # Colorbar
    f.colorbar(m,loc='b',length=0.75)
    
    # Save to file
    plt.savefig(figfile)
    

def testModelsCat(ds_features,ds_target,fname):
    
    # Setup Features (X) and Target (Y)
    
    X=ds_features.to_stacked_array('features',sample_dims=['time'])
    Y=ds_target['precip'].values

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
    
    # Train the Models
        
    # -- Logistic Regression -- #
    regr_log,Ypred_log=logistic(X_train,Y_train)
    print('Logistic Training set accuracy score: ' + str(regr_log.score(X_train,Y_train)))
    print('Logistic Test set accuracy score: ' + str(regr_log.score(X_test,Y_test)))
    
    # -- Neural Network -- #
    weights = class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train)
    nn = KerasClassifier(build_fn=tomsensomodel_cat(X_train.shape[1]),epochs=100, batch_size=25,verbose=0,class_weight=weights)
    history=nn.fit(X, Y,validation_split=0.2,shuffle=False)
    Ypred_nn=nn.predict(X_train)
    
    print("CHECK NN: ",np.count_nonzero(Ypred_nn==0),np.count_nonzero(Ypred_nn==1))
    print('NN Training set accuracy score: ' + str(nn.score(X_train, Y_train)))
    print('NN Test set accuracy score: ' + str(nn.score(X_test, Y_test)))
    
    # Plot learning Curve for NN
    plt.figure(figsize=(11,8.5))
    plotLearningCurve(history)
    
    # Plot training and target
    plt.figure(figsize=(11,8.5))
    plt.step(np.arange(ntrain),Y_train)
    plt.step(np.arange(ntrain),Ypred_log)
    plt.step(np.arange(ntrain),Ypred_nn)
    plt.legend(['Target','Logistic','NN'])
    
    # Plot Coefficients for Logistic Regression
    plt.figure(figsize=(11,8.5))
    plt.bar(list(ds_features.keys()),regr_log.coef_[0],edgecolor='k',facecolor='b')
    plt.title('Logistic Regression')
    plt.xlabel('Predictor')
    plt.ylabel('Regression Coefficient')
    plt.xticks(np.arange(nfeatures),list(ds_features.keys()),rotation=45)
    plt.savefig('../figs/'+fname+'.logistic_coeffs.png')
    
    # Calculate LRP 
    a=calcLRP(nn.model,X_train)
    ds_tmp=xr.DataArray(a,
                    coords={'time':ds_features['time'][0:a.shape[0]],
                            'features':list(ds_features.keys())},
                    dims=['time','features'])        
    ds_lrp=ds_tmp.to_dataset(name='lrp')

    # Normalize by max value in grid
    ds_lrp=ds_lrp/ds_lrp.max(dim=['features'])
    
    # Plot LRP composite in time      
    plt.figure(figsize=(11,8.5))
    plt.bar(ds_lrp['features'],ds_lrp['lrp'].mean(dim='time'),edgecolor='k',facecolor='b')  
    plt.title('Shallow Neural Network')
    plt.ylabel('Normalized Relevance (unitless)')
    plt.xlabel('Predictor')
    plt.xticks(np.arange(nfeatures),list(ds_features.keys()),rotation=45)
    plt.savefig('../figs/'+fname+'.nnlrp.png')
    
    return

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
    nn = KerasRegressor(build_fn=tomsensomodel_regression(X_train.shape[1]),epochs=250, batch_size=100,verbose=0)
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

def getMonthlyClimIndices(file,i,sdate,edate):

    df=pd.read_table(file,skiprows=1,
                     header=None,delim_whitespace=True,
                     index_col=0,parse_dates=True,
                     na_values=['-99.9','-99.90','-99.99']).dropna()
    
    start_date=str(df.index[0])+'-'+str(df.columns[0])+'-01'
    end_date=str(df.index[-1])+'-'+str(df.columns[-1])+'-01'
    dates=pd.date_range(start=start_date,end=end_date,freq='MS') + pd.DateOffset(days=14)
    
    ds=xr.DataArray(df.T.unstack().values.astype('float'),coords={'time':dates},dims=['time']).to_dataset(name=i).dropna(dim='time')
    
    return ds

def getRMM(file,sdate,edate):

    rmm_cols=['year','month','day','rmm1','rmm2','phase','amp','source'] 

    df=pd.read_table(file,skiprows=2,
                     header=None,delim_whitespace=True,
                     names=rmm_cols,parse_dates= {"date" : ["year","month","day"]},
                     na_values=['999','1e36']).dropna().drop(['source'],axis=1)
    ds_phase=xr.DataArray(df['phase'].astype(int)-1,coords={'time':df['date']},dims=['time']).to_dataset(name='RMM_phase').sel(time=slice(sdate,edate)) 
    ds_phase['RMM_phase_bins']=np.arange(9)
    ds_amp=xr.DataArray(df['amp'],coords={'time':df['date']},dims=['time']).to_dataset(name='RMM_amp').sel(time=slice(sdate,edate))        
   
    return ds_phase,ds_amp

def getWR(file,seas,sdate,edate):
    fname=file+seas+'.nc'
    ds=xr.open_dataset(fname).rename({'clusters':'pnaregimes'}).sel(time=slice(sdate,edate))
    ds['pnaregimes_bins']=np.arange(5)
    return ds

def getMLSO(file,sdate,edate):
    ds=xr.open_dataset(file).sel(time=slice(sdate,edate))
    return ds

def getNASH(file,sdate,edate):
    
    ds=xr.open_dataset(file)

    ds_amp=ds['amp'].to_dataset(name='nash_amp')
    
    ds_phase=ds['phase'].to_dataset(name='nash_phase')
    ds_phase['nash_phase_bins']=np.arange(4)
    
    return ds_phase,ds_amp

def plotLearningCurve(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return


def calcLRP(model,X):
    
    # Strip softmax layer
    #model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.create_analyzer("deep_taylor", model)

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(X)
    
    return a

def testModelsCatField(ds_features,ds_target):

    
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
    nfeatures=X.shape[1]

    print("Samples: ",nsamples)
    print("Features: ", nfeatures)
    
    # Create Train and Test Sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,shuffle=False)

    ntrain=X_train.shape[0]
    ntest=X_test.shape[0]

    print('Training Size: ',ntrain)
    print('Testing Size: ',ntest)
    
    # Train the Models
        
    # -- Logistic Regression -- #
    #regr_log,Ypred_log=logistic(X_train,Y_train)
    #print('Logistic Training set accuracy score: ' + str(regr_log.score(X_train,Y_train)))
    #print('Logistic Test set accuracy score: ' + str(regr_log.score(X_test,Y_test)))
    
    # -- Neural Network -- #
    weights = class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train)
    nn = KerasClassifier(build_fn=cnn_cat(X_train.shape[1:]),epochs=100, batch_size=25,verbose=0,class_weight=weights)
    history=nn.fit(X, Y,validation_split=0.2,shuffle=False)
    Ypred_nn=nn.predict(X_train)
    
    print("CHECK NN: ",np.count_nonzero(Ypred_nn==0),np.count_nonzero(Ypred_nn==1))
    print('NN Training set accuracy score: ' + str(nn.score(X_train, Y_train)))
    print('NN Test set accuracy score: ' + str(nn.score(X_test, Y_test)))
    
    # Plot learning Curve for NN
    plt.figure(figsize=(11,8.5))
    plotLearningCurve(history)
    
    # Plot training and target
    plt.figure(figsize=(11,8.5))
    plt.step(np.arange(ntrain),Y_train)
    #plt.step(np.arange(ntrain),Ypred_log)
    plt.step(np.arange(ntrain),Ypred_nn)
    plt.legend(['Target','NN'])
    
    # Plot Coefficients for Logistic Regression
    #plt.figure(figsize=(11,8.5))
    #coeffs_2d=regr_log.coef_[0].reshape(ny,nx)
    #plt.contourf(ds_features['lon'],ds_features['lat'],coeffs_2d)
    #plt.colorbar()
    
    # Calculate LRP 
    a=calcLRP(nn.model,X_train)
    ds_tmp=xr.DataArray(a.reshape(ntrain,ny,nx,nvar),
                    coords={'time':ds_features['time'][0:ntrain],
                            'lat':ds_features['lat'],
                            'lon':ds_features['lon'],
                            'var':['z500','u250']},
                    dims=['time','lat','lon','var'])        
    ds_lrp=ds_tmp.to_dataset(name='lrp')
    
    # Normalize by max value in grid
    ds_lrp=ds_lrp/ds_lrp.max(dim=['lat','lon'])
    
    # Plot LRP composite in time    
    plotLRP(ds_lrp)
    
    return


def plotLRP(ds):

    # Define map region and center longitude
    lonreg=(ds['lon'].values.min(),ds['lon'].values.max())
    latreg=(ds['lat'].values.min(),ds['lat'].values.max())
    lon_0=290
    
    print(lonreg,latreg,lon_0)
    
    # Plotting
    cmap='Blues'
    clevs=np.arange(0.0,0.5,.01)
    
    # Variables
    vnames=ds['var'].values
    nvars=len(vnames)
    
    f, axs = pplt.subplots(ncols=1, nrows=nvars,
                           proj='pcarree',proj_kw={'lon_0': lon_0},
                           width=8.5,height=11.0)
    
    for i,v in enumerate(vnames):

        ds_tmp=ds['lrp'].sel(var=v).mean(dim='time')
        
        m=axs[i].contourf(ds['lon'], ds['lat'],
                          ds_tmp,levels=clevs,
                          cmap=cmap,extend='both')
        axs[i].format(coast=True,lonlim=lonreg,latlim=latreg,grid=True,
                      lonlabels='b', latlabels='l',title=v,
                      suptitle='LRP Relevance',abc=True)
        # Add US state borders    
        axs[i].add_feature(cfeature.STATES,edgecolor='gray')

    # Colorbar
    f.colorbar(m,loc='b',length=0.75)
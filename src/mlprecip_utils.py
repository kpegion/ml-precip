import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression

import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import optimizers

import proplot as pplt
import cartopy.feature as cfeature

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
    Y_pred = regr.predict(X)

    r_squared,Y_pred=get_r2(X,Y,regr)

    return regr,regr.coef_,r_squared,Y_pred

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
    Y_pred = regr.predict(X)

    r_squared,Y_pred=get_r2(X,Y,regr)

    return regr,regr.coef_,r_squared,Y_pred

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
    Y_pred = regr.predict(X)

    r_squared,Y_pred=get_r2(X,Y,regr)

    return regr,regr.coef_,r_squared,Y_pred
def tomsensomodel_regression(X,Y):

    """
    Fit fully connected neural network Input(nfeatures)->8->8->Output(1)

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)

    Returns:
    model

    """
    model = Sequential()

    model.add(Dense(8, input_dim=X.shape[1],activation='tanh',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l1(0.02),
                bias_initializer='he_normal'))

    model.add(Dense(8, activation='tanh',
                kernel_initializer='he_normal',
                bias_initializer='he_normal'))

    model.add(Dense(1,name='output'))

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss ='mean_squared_error',
                  metrics = ['mse'])

    model.fit(X,Y,epochs=250, batch_size=100,verbose=0)

    return(model)

def get_r2(X,Y,model):

    """
    Calculate r-squared of a for a given model, features, and target

    Args:
    X : numpy array representing the features for all nsamples of the training data  (nsamples,nfeatures)
    Y : numpy array representing the target for all nsamples of the training data (nsamples)
    model : model returned from calls to keras or scikit-learn models

    Returns:
    r-squared value of predicted value of Y given X and target value of Y based on specified model

    """
    pred = model.predict(X).squeeze()
    rsq=np.corrcoef(Y,pred)[0,1]
    return rsq,pred


def standardize(ds):
    """
    Standardize the dataset as (X-mu)/sigma

    Args:
    ds : xarray.Dataset with dimensions time, ..., ...

    Returns:
    Standardized xarray.Dataset

    """
    ds_scaled=(ds-ds.mean(dim='time'))/ds.std(dim='time')
    return ds_scaled

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
    d = pd.DataFrame(data=tmp,columns=labels)
    corr=d.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(11,11))
    sns.set(font_scale=1)
    ax=sns.heatmap(corr,square=True,linewidths=0.5,fmt=" .2f", \
                   annot=True,mask=mask,cmap='seismic', \
                   vmin=-1,vmax=1,cbar=False,annot_kws={"size": 10})
    
def makeBins(ds,index_name,nbins):

    # Create bins and assign integer to each bin
    tmp=np.zeros((ds['time'].shape[0]))
    tmp[ds[index_name]>=0.5]=0
    tmp[(ds[index_name]>=0) & (ds[index_name]< 0.5)] = 1
    tmp[ds[index_name]<=-0.5]=2

    # Put into xarray.Dataset
    ds_tmp=xr.DataArray(tmp,
                        coords={'time':ds['time'].values},
                                dims=['time'])        
    ds_tmp=ds_tmp.to_dataset(name=index_name+'_bins')
    
    return(ds_tmp)
    
def calcComposites(ds,index_name,labels):
    
    totals=[]
    
    # Determine number of bins
    nbins=len(ds[index_name+'_bins'])
    
    # Loop over bins to calculate number in each bin and print
    for j,l in zip(range(nbins),labels):

        total=ds[index_name+'_bins'].where(ds[index_name+'_bins']==j).count(dim='time')
        print(l,total.values)
        totals.append(total)

    # Calculate the composites
    ds_comp_anoms=ds.groupby(index_name+'_bins').mean(dim='time')

    return ds_comp_anoms,totals

def plotComposites(ds,index_name,totals,suptitle,labels,clevs,cmap,figfile):
    
  
    # Define map region and center longitude
    lonreg=(269,283)
    latreg=(24,36)
    lon_0=290

    f, axs = pplt.subplots(ncols=1, nrows=3,
                           proj='pcarree',proj_kw={'lon_0': lon_0},
                           width=8.5,height=11.0)
    dim_str=index_name+'_bins'
    nbins=len(ds[dim_str])
    
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
    
    return ds

def calcRatios(ds,index_name,v,thresh):
    
    above=(ds[v].where(ds[v]>thresh)).groupby(ds[index_name+'_bins']).count(dim='time')
    below=(ds[v].where(ds[v]<thresh)).groupby(ds[index_name+'_bins']).count(dim='time')
    ratio=above/below 
    
    return above,below,ratio

def plotRatios(da,index_name,suptitle,labels,clevs,cmap,figfile):

    # Define map region and center longitude
    lonreg=(269,283)
    latreg=(24,36)
    lon_0=290

    f, axs = pplt.subplots(ncols=1, nrows=3,
                           proj='pcarree',proj_kw={'lon_0': lon_0},
                           width=8.5,height=11.0)

    dim_str=index_name+'_bins'
    nbins=len(da[dim_str])
    
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
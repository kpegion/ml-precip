import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da

from keras.utils import np_utils

import proplot as pplt
import cartopy.feature as cfeature
import xskillscore as xs

import innvestigate
import innvestigate.utils

# Turn off deprecation warnings
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    
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

def plotLearningCurve(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return

def padforcnn(img_list,in_shape):
    
    # From https://stackoverflow.com/questions/50022256/keras-2d-padding-and-input
    desiredX = in_shape[0]
    desiredY = in_shape[1]
    
    padded_images = []

    for img in img_list:
        shape = img.shape
        xDiff = desiredX - shape[0]
        xLeft = xDiff//2
        xRight = xDiff-xLeft

        yDiff = desiredY - shape[1]
        yLeft = yDiff//2
        yRight = yDiff - yLeft

        padded_images.append(np.pad(img,((xLeft,xRight),(yLeft,yRight),(0,0)), mode='constant'))

    padded_images = np.asarray(padded_images)
                             
    return padded_images 

def make_ohe_thresh_terc(da):
    
    # One hot encoding for target by upper and lower thresholds
    
    thresh_upper=np.percentile(da,66)
    thresh_lower=np.percentile(da,33)
   
    tmp=xr.where(da>=thresh_upper,2,da)
    tmp=xr.where(da<=thresh_lower,0,tmp)
    tmp=xr.where(np.logical_and(da>thresh_lower,da<thresh_upper),1,tmp)
    
    print(np.count_nonzero(tmp==2))
    print(np.count_nonzero(tmp==1))
    print(np.count_nonzero(tmp==0))
    
    enc = np_utils.to_categorical(tmp.values)
    
    return enc

def make_ohe_thresh_med(da):
    
    # One hot encoding for target by upper and lower thresholds
    
    #thresh=np.percentile(da,50)
    #print(thresh)
    thresh=0.0
   
    tmp=xr.where(da>=thresh,1,0)
    
    print("Upper Cat: ",np.count_nonzero(tmp==1))
    print("Lower Cat: ",np.count_nonzero(tmp==0))
    
    enc = np_utils.to_categorical(tmp.values)
    
    return enc


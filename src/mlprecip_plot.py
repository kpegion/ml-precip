import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da

import proplot as pplt
import cartopy.feature as cfeature
import xskillscore as xs

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

def plot_reliability(ds_model,cat_labels):
    
    
    f, axs = pplt.subplots(ncols=len(cat_labels),nrows=1)
    
    rel_bins=np.arange(0,1.1,0.1)

    for iplot,(ax,cat) in enumerate(zip(axs,cat_labels)):
        
        rel=xs.reliability(ds_model['verif']==iplot,ds_model['probs'].sel(cat=cat),dim=['model','time'],probability_bin_edges=rel_bins)
        
        ax.plot(rel['forecast_probability'],rel,'b')
        ax.plot(rel['forecast_probability'],rel,'b.')
        ax.format(xtickminor=True,ytickminor=True,title=cat,xlabel='Forecast Probability',ylabel='Observed Probability')
        ax.plot(rel_bins,rel_bins,'k')
        axi = ax.inset([0.125, 0.8, 0.4, 0.2], transform='data',zoom=False)
        axi.bar(rel['samples']/rel['samples'].sum(dim='forecast_probability'))
        axi.format(xtickminor=True,ytickminor=True,fontsize=6)
    
    return
   
def plotLRP_bar(x,y1,y2,fname='',title='Neural Network'):
    
    plt.figure(figsize=(11,8.5))
    plt.bar(x,y1,yerr=y2,edgecolor='k',facecolor='b')  
    plt.title(title)
    plt.ylabel('Relevance (unitless)')
    plt.xlabel('Predictor')
    plt.xticks(np.arange(len(x)),x,rotation=45)
    #if (fname):
       #plt.savefig('../figs/'+fname+'.nnlrp.png'))

    return



def plotLRP_map(ds,mean_dim):

    # Define map region and center longitude
    lonreg=(ds['lon'].values.min(),ds['lon'].values.max())
    latreg=(ds['lat'].values.min(),ds['lat'].values.max())
    lon_0=290
    
    # Plotting
    cmap='fire'
    clevs=np.arange(0.0,0.3,.005)
    
    # Variables
    vnames=ds['var'].values
    nvars=len(vnames)
    
    f, axs = pplt.subplots(ncols=1, nrows=nvars,
                           proj='pcarree',proj_kw={'lon_0': lon_0},
                           width=8.5,height=11.0)
    
    for i,v in enumerate(vnames):

        ds_tmp=ds['lrp'].sel(var=v).mean(dim=mean_dim)
        
        m=axs[i].contourf(ds['lon'], ds['lat'],
                          ds_tmp,levels=clevs,
                          cmap=cmap,extend='both')
        axs[i].format(coast=True,lonlim=lonreg,latlim=latreg,grid=True,
                      lonlabels='b', latlabels='l',title=v,
                      suptitle='Mean LRP Relevance',abc=True)
        
        # Add US state borders    
        axs[i].add_feature(cfeature.STATES,edgecolor='gray')

    # Colorbar
    f.colorbar(m,loc='b',length=0.75)
    
def plotTarget_med(da_target,da_pred,da_verif,thresh,fname=''):
    
    plt.figure(figsize=(11,8.5))

    # Plot gray where fcsts are not correct
    tmp=da_target.where(da_pred!=da_verif)
    plt.plot(tmp['time'],tmp,color='gray',marker='.',markersize=8,linestyle='None')

    # Plot correct above median forecatsts in green
    tmp=da_target.where(np.logical_and(da_pred==da_verif,da_target>=thresh))
    plt.plot(tmp['time'],tmp,'g.',markersize=8,linestyle='None')

    # Plot correct below median forecasts in brown
    tmp=da_target.where(np.logical_and(da_pred==da_verif,da_target<thresh))
    plt.plot(tmp['time'],tmp,color='brown',marker='.',markersize=8,linestyle='None')

    # Draw median line
    plt.axhline(thresh,color='k')
    
    # Highlight test data
    plt.axvspan(2403, 3003, color='gray', alpha=0.5)
    
    # Save figure to fname
    
    return
    
def plot_modelaccHist(acc):

    plt.figure(figsize=(11,8.5))
    hist_bins=np.arange(40,100,5)
    plt.hist(acc*100,hist_bins,color='b',edgecolor='k')

    

        
    
    

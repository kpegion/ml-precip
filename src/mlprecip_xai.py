import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import proplot as pplt
import cartopy.feature as cfeature

import innvestigate
import innvestigate.utils

from mlprecip_utils import *

def calcLRP(model,X,rules=['lrp.z']):
    
    # Strip softmax layer
    modelwosf = innvestigate.utils.model_wo_softmax(model)
    
    a_list=[]
    for rule in rules:

        # Create analyzer
        analyzer = innvestigate.create_analyzer(rule, modelwosf)

        # Apply analyzer w.r.t. maximum activated output-neuron
        a = analyzer.analyze(X)
    
        a_list.append(a)
        
    return a_list
    
def LRPClusters(ds,dim_name):

    cmap='fire'
    clevs=np.arange(0.0,1.0,.05)
    nclusters=2
    nx=len(ds['lon'])
    ny=len(ds['lat'])
    nt=len(ds['time'])
    
    vnames=ds['var'].values
    nvars=len(vnames)
    
    # Define map region and center longitude
    lonreg=(ds['lon'].values.min(),ds['lon'].values.max())
    latreg=(ds['lat'].values.min(),ds['lat'].values.max())
    lon_0=290

    f, axs = pplt.subplots(ncols=2, nrows=nvars,
                        proj='pcarree',proj_kw={'lon_0': lon_0},
                        width=11.0,height=8.5)
    
    for i,v in enumerate(vnames):
        
        kmeans = KMeans(n_clusters=nclusters,random_state=1,init='random',n_init=100)
        tmp=(ds['lrp'].sel(var=v).values).reshape(nt,nx*ny)
        kmeans.fit(tmp)
        y=kmeans.predict(tmp)
        ds[dim_name]=y

        cluster_comp=ds.sel(var=v).groupby(dim_name).mean()
    
        cluster_freq=(ds.sel(var=v).groupby(dim_name).count())/nt
        #print(cluster_freq.values)
        
        for j in range(nclusters):
            m=axs[i,j].contourf(ds['lon'], ds['lat'],cluster_comp['lrp'][j,:,:],levels=clevs,extend='both',cmap=cmap)
            axs[i,j].format(coast=True,lonlim=lonreg,latlim=latreg,grid=True,
                        lonlabels='b', latlabels='l',title=v+' Cluster: '+str(j))
    
            # Add US state borders    
            axs[i,j].add_feature(cfeature.STATES,edgecolor='gray')

    # Colorbar
    f.colorbar(m,loc='b',length=0.75)    
    
    return 
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

def init_predictors_dict():
    
    amo_dict=dict(name='amo',ptype='index',freq='mon',readfunc='getMonthlyClimIndices')
    naomonthly_dict=dict(name='nao',ptype='index',freq='mon',readfunc='getMonthlyClimIndices')
    nino34_dict=dict(name='nino34',ptype='index',freq='mon',readfunc='getMonthlyClimIndices')
    pdo_dict=dict(name='pdo',ptype='index',freq='mon',readfunc='getMonthlyClimIndices')
    rmmamp_dict=dict(name='RMM_amp',ptype='index',freq='day',readfunc='getRMM')
    rmmphase_dict=dict(name='RMM_phase',ptype='cat',freq='day',readfunc='getRMM')

    predictors=[amo_dict,naomonthly_dict,nino34_dict,pdo_dict,rmmamp_dict,rmmphase_dict]
    
    return predictors
    

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
    
def makeCategories(ds,bins,index_name):

    # This is more generic than makeBins
    
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

def makeBins(ds,index_name,nbins):

    # Create bins and assign integer to each bin
    tmp=np.zeros((ds['time'].shape[0]))
    tmp[ds[index_name]>=0.5]=0
    tmp[(ds[index_name]>=-0.5) & (ds[index_name]< 0.5)] = 1
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

    # Set number of rows and columns
    if (nbins > 4):
        ncols=2
        nrows=int(nbins/ncols)
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

    # Set number of rows and columns
    if (nbins > 4):
        ncols=2
        nrows=int(nbins/ncols)
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
    
def getMonthlyClimIndices(path,i,sdate,edate):

    print(i)
    df=pd.read_table(path+'/CLIM_INDICES/'+i+'.txt',skiprows=1,
                     header=None,delim_whitespace=True,
                     index_col=0,parse_dates=True,
                     na_values=['-99.9','-99.90','-99.99']).dropna()
    
    start_date=str(df.index[0])+'-'+str(df.columns[0])+'-01'
    end_date=str(df.index[-1])+'-'+str(df.columns[-1])+'-01'
    dates=pd.date_range(start=start_date,end=end_date,freq='MS') + pd.DateOffset(days=14)
    
    ds=xr.DataArray(df.T.unstack().values.astype('float'),coords={'time':dates},dims=['time']).to_dataset(name=i).dropna(dim='time')

    # Linearly interpolate monthly indices to daily
    ds=ds.resample(time='1D').interpolate("linear").sel(time=slice(sdate,edate))
    
    return ds

def getRMM(path,sdate,edate):

    rmm_cols=['year','month','day','rmm1','rmm2','phase','amp','source'] 
    file='rmmint1979-092021.txt'

    df=pd.read_table(path+'/RMM/'+file,skiprows=2,
                     header=None,delim_whitespace=True,
                     names=rmm_cols,parse_dates= {"date" : ["year","month","day"]},
                     na_values=['999','1e36']).dropna().drop(['source'],axis=1)
    ds_phase=xr.DataArray(df['phase'].astype(int)-1,coords={'time':df['date']},dims=['time']).to_dataset(name='RMM_phase').sel(time=slice(sdate,edate)) 
    ds_phase['RMM_phase_bins']=np.arange(9)
    ds_amp=xr.DataArray(df['amp'],coords={'time':df['date']},dims=['time']).to_dataset(name='RMM_amp').sel(time=slice(sdate,edate))        
   
    return ds_phase,ds_amp

def testModels(ds_i):
    
    # ML Model for this season
    
    # Setup Features (X) and Target (Y)
    # Features: AMO, NAO, Nino34, PDO; Target: SEUS Precip Index
    
    X=np.stack((ds_i['amo'].values,ds_i['nao'].values,ds_i['nino34'],ds_i['pdo']),axis=-1)
    Y=ds_i['precip'].values

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
    
    # Take a look at the Training Data
    
    plt.figure(figsize=(11,8.5))
    y=np.arange(ntrain)

    for i,f in enumerate(indices):

        plt.subplot(2,2,i+1)

        z = np.polyfit(y,X_train[:,i],1)
        p = np.poly1d(z)
    
        plt.plot(y,X_train[:,i])
        plt.plot(p(y),"r--")
        plt.title(f)

        print("Check Stats: ", "Index: ",f, "Mean: ", X_train[:,i].mean(axis=0),"Var: ", X_train[:,i].var(axis=0))
    plt.tight_layout()  
    
    # Make a heatmap
    heatmap(X_train,Y_train,list(ds_i.keys()))
    
    # Train the Models
    regr_lr,coeffs_lr,rsq_train_lr,Ypred_lr=lr(X_train,Y_train)
    print('R^2 Train Standard : ', rsq_train_lr)
    regr_lasso,coeffs_lasso,rsq_train_lasso,Ypred_lasso=lasso(X_train,Y_train)
    print('R^2 Train LASSO : ', rsq_train_lasso)
    regr_ridge,coeffs_ridge,rsq_train_ridge,Ypred_ridge=ridge(X_train,Y_train)
    print('R^2 Train Ridge : ', rsq_train_ridge)
    nn=tomsensomodel_regression(X_train,Y_train)
    rsq_train_nn,Y_pred_train_nn=get_r2(X_train,Y_train,nn)
    print('R^2 Train NN: ',rsq_train_nn)
    
    # Predict for Test
    rsq_test_lr,Y_pred_test_lr=get_r2(X_test,Y_test,regr_lr)
    print('R^2 Test Standard: ',rsq_test_lr)
    rsq_test_lasso,Y_pred_test_lasso=get_r2(X_test,Y_test,regr_lasso)
    print('R^2 Test Lasso: ',rsq_test_lasso)
    rsq_test_ridge,Y_pred_test_ridge=get_r2(X_test,Y_test,regr_ridge)
    print('R^2 Test Ridge: ',rsq_test_lr)
    rsq_test_nn,Y_pred_test_nn=get_r2(X_test,Y_test,nn)
    print('R^2 Test NN: ',rsq_test_nn)
    
    # Plot Coefficients for Standard Linear Regression
    plt.figure(figsize=(11,8.5))
    plt.bar(indices,coeffs_lr)

    return




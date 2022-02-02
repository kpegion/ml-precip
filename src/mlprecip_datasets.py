import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da


def init_predictors_dict():
    
    amo_dict=dict(name='amo',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                 file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/amo.txt',seas=['DJF','JJA'])
    naomonthly_dict=dict(name='nao',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                        file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/nao.txt',seas=['DJF','JJA'])
    nino34_dict=dict(name='nino34',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                    file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/nino34.txt',seas=['DJF','JJA'])
    pdo_dict=dict(name='pdo',ptype='index',freq='mon',readfunc='getMonthlyClimIndices',
                 file='/data/ccsm4/kpegion/obs2/CLIM_INDICES/pdo.txt',seas=['DJF','JJA'])
    rmmamp_dict=dict(name='RMM_amp',ptype='index',freq='day',readfunc='getRMM',
                     file='/data/ccsm4/kpegion/obs2/RMM/rmmint1979-092021.txt',seas=['DJF','JJA'])
    rmmphase_dict=dict(name='RMM_phase',ptype='cat',freq='day',readfunc='getRMM',
                      file='/data/ccsm4/kpegion/obs2/RMM/rmmint1979-092021.txt',seas=['DJF','JJA'])
    pnaregimes_dict=dict(name='pnaregimes',ptype='cat',freq='day',readfunc='getWR',
                         file='/scratch/kpegion/ERAI_clusters_5_1980-2015_',seas=['DJF','JJA'])
    mlso_dict=dict(name='mlso',ptype='index',freq='day',readfunc='getMLSO',
                  file='/data/vortex/scratch/mlso.index.01011979-08312019.nc',seas=['DJF','JJA'])
    nashamp_dict=dict(name='nash_amp',ptype='index',freq='day',readfunc='getNASH',
                      file='/scratch/kpegion/ERAI_NASH_JJA.1979-2019.nc',seas=['JJA'])
    nashphase_dict=dict(name='nash_phase',ptype='cat',freq='day',readfunc='getNASH',
                        file='/scratch/kpegion/ERAI_NASH_JJA.1979-2019.nc',seas=['JJA'])
    z500local_dict=dict(name='z500_local',ptype='index',freq='day',readfunc='getz500_local',
                        file='/project/predictability/kpegion/wxregimes/era-interim/erai_z500_1979-2019.nc',
                        seas=['DJF','JJA'])


#    predictors=[amo_dict,naomonthly_dict,nino34_dict,pdo_dict,rmmamp_dict,rmmphase_dict,mlso_dict,pnaregimes_dict,nashphase_dict,nashamp_dict]
    predictors= [amo_dict,naomonthly_dict,nino34_dict,pdo_dict,rmmamp_dict,rmmphase_dict,mlso_dict,nashphase_dict,nashamp_dict,pnaregimes_dict]

    return predictors


#def init_fields_dict():
#    
#    pna_z500_dict=dict(name='pnaz500',freq='day',readfunc='getfield',
#                       file='/scratch/kpegion/ERAI_clusters_5_1980-2015_')
#    global_z500=dict(name='globalz500,freq='day', readunc='getfield',
#                     file='/project/predictability/kpegion/wxregimes/era-interim/erai_z500_1979-2019.nc'


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

def getz500_local(file,lat_reg,lon_reg,sdate,edate):
    
    ds_tmp=xr.open_dataset(file)

    ds_tmp=ds_tmp['z500'].to_dataset(name='z500_local')
    
    ds_anoms=ds_tmp.groupby('time.dayofyear')-ds_tmp.groupby('time.dayofyear').mean()
    ds=ds_anoms.sel(lat=slice(lat_reg[1],lat_reg[0]),lon=slice(lon_reg[0],lon_reg[1])).mean(dim=['lat','lon'])
    print(ds)
    
    return ds


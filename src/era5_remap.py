#!/usr/bin/env python
# coding: utf-8

# ## Reconstitute reduced-grid ERA5 data onto a lat-lon grid
# 
# #### Background
# 
# ##### Reduced Gaussian grid
# 
# ERA5 is a ~31 km global data set of many 2-D and 3-D variables spanning decades. The native data output is on the **N320** reduced Gaussian grid. On a regular latitude-longitude grid, there would be 1280 grid cells in longitude (9/32˚ spacing), 640 in latitude (Gaussian grid spacing of ~0.28˚, slightly closer spacing near the poles, wider at the equator). The reduced grid thins points on each of the 640 latitude circles to maintain approximately 32 km grid spacing, also saving some storage space. As a result, each global 2-D horizontal grid is stored as 1-D arrays on a dimension called `rgrid` of length 542080, saving more than 1/3 of the space. Atmospheric 3-D fields on pressure levels (not the complete set available) can be found at: `/shared/land/ERA5/daily/global_3d/`
# 
# Complete georegistration data to remap these fields is not contained in the original data files. Only the Gaussian latitudes are included in the ERA5 data files - the corresponding longitudes for each point are not. There is a table with this information (see: https://confluence.ecmwf.int/display/USS/Gaussian+grid+number+320 or the comma-sparated-value file `/shared/land/ERA5/Reduced_Gaussian.csv`), where the first point in each *row* at a given Gaussian latitude is always located at longitude 0˚E. Such data can be plotted directly in Python using, for example, the `matplotlib.pyplot` functions `scatter` or `hexbin`, but the gaps may show at higher latiudes, depending on the projection. The function below uses the *csv* file above to reconstitute the full Gaussian grid using a nearest-neighbor approach along each latitude band.
# 
# ##### Land-only data
# 
# There is also a great deal of 2-D daily global ERA5 data processed to retain only fields over over land (excluding Antarctica) at: `/shared/land/ERA5/daily/global/`. To save space, the data have been stored as 1-D series only over land points on the original N320 reduced Gaussian grid (~31 km resolution). This is like what was done for GSWP2, and is jokingly referred to as "dehydrated" since all the water has been removed. The files `compressed_sfc_fc_daily_*.nc4` each contain one month's worth of daily output for 36 variables plus static fields of latitude and longitude for each grid cell. These data are saved as 1-D arrays on a dimension called `lgrid` of length 141780, a compression of nearly 83% over the full global grid.
# 
# As with the global reduced Gaussian data described above, there is also a separate utility file for the land-only data: `/shared/land/ERA5/land_index.nc`, which is a 2-D array representing the original georegistered grid: 1280 grid cells in longitude and 534 cells in latitude (60˚S to the North Pole). The file `land_index.nc` is used to "uncompress" the other data by remapping the contents of each 1-D array element to one or more cells on the 2-D grid.
# 
# ----------
# 
# The Python code below will decompress the data back into geo-registered (640 in latitude by 1280 in longitude) 2-D grids, following the same "nearest neighbor" interpolation in longitude that is used by ECMWF.
# 
# 
# #### `era5_remap(var,type)`
# 
# The function `era5_remap` takes two arguments: 
# 1. `var` is the name of the reduced grid *xarray* DataArray to be expanded back to a full rectangular grid. It can be a data variable at one time (a 1-D array along `lgrid`), a time series (`time` and `lgrid` dimensions), a nominal 3-D grid (`lgrid` and vertical dimensions) or a time series of 3-D grids. 
# 2. `type` is a string that indicates the type of reduced data:
#     * `"rgg"` indicates global data on the reduced Gaussian grid
#     * `"land`" indicates a land-only data 
# 
# The function will not work on an entrire *xarray* Dataset, only on individual DataArrays (i.e., one variable at a time).
# 
# 
# #### Package requirements
# 
# Only `xarray` and `numpy` are needed.
# 
# 
# #### Performance
# 
# On **atlas3** it takes about 1/4 second to uncompress one land-only 1-D array into a 2-D *xarray* DataArray. To uncompress a variable for an entire month takes about 10 seconds.


import numpy as np
import xarray as xr



def era5_remap(var,type):   # Uncompressor as a defined function
    """
    `var` is the name of the reduced grid xarray DataArray to be expanded back to a full rectangular grid. `var` can be:
        For land-only grids (type=="land")
        • a data variable at one time (a 1-D array along `lgrid`)
        • a time series (`time` and `lgrid` dimensions)
        For global reduced Gaussian grids (type=="rgg")
        • a data variable at one time (a 1-D array along `rgrid`)
        • a time series (`time` and `rgrid` dimensions)
        • a 3-D grid (`rgrid` and vertical `plev` dimensions) 
        • a time series of 3-D grids. 
        
    `type` is a string that indicates the type of reduced data:
        • "rgg" indicates global data on the reduced Gaussian grid
        • "land" indicates land-only data 
    """
    
########################################    
    if type == "land":  # Land-only data
        if 'l_pdef' not in globals():
            l_pdef = xr.open_dataset('/shared/land/ERA5/land_index.nc')     # This global Dataset has the index grid to reconstitute the 2D grids:

        igrid = np.nan_to_num(l_pdef.index.values).astype(int)               # Need the index grid to be a flat integer list (0 = not land)
        tmp = np.insert(var,0,np.nan,axis=var.dims.index('lgrid')).values    # Compressed series needs a 'nan' inserted at index 0 to mask around land cells

        if var.ndim == 1:                                                               # If there is no varying time dimension...
            uuu = [[tmp[i] for i in j] for j in igrid]                                  # Result is a 2D list [lat,lon]
            res = l_pdef.index.copy(deep=True, data=np.asarray(uuu)).rename(var.name)   # Put back into an xarray for output

        else:                                                                # If there is also a time dimension to work across...
            uuu = [[[tmp[k,i] for i in j] for j in igrid] for k in range(tmp.shape[0])] # Result is a 3D list [time,lat,lon]
            res = l_pdef.expand_dims({'time':var.time},axis=0).index.copy(deep=True, 
                                data=np.asarray(uuu)).rename(var.name)       # xarray with proper time dimension too

########################################    
    if type == "rgg":   # Global reduced Gaussian grid
        if 'r_pdef' not in globals():
            r_pdef = xr.open_dataset('/shared/land/ERA5/N320_index.nc')      # This global Dataset has the index grid to reconstitute the 2D grids:

        igrid = np.nan_to_num(r_pdef.index.values)                           # Need the index grid to be a flat integer list
        tmp = var.values                                                     # Compressed series needs a 'nan' inserted at index 0 to mask around land cells

        if var.ndim == 1:     # If there is no varying time or p-level dimension...
            uuu = [[tmp[i] for i in j] for j in igrid]                                  # Result is a 2D list [lat,lon]
            res = r_pdef.index.copy(deep=True, data=np.asarray(uuu)).rename(var.name)   # Put back into an xarray for output

        if var.ndim == 2:     # If there is also one other varying dimension...
            if 'time' in var.dims:                                                          # If there is also a time dimension to work across...
                uuu = [[[tmp[k,i] for i in j] for j in igrid] for k in range(tmp.shape[0])] # Result is a 3D list [time,lat,lon]
                res = r_pdef.expand_dims({'time':var.time},axis=0).index.copy(deep=True, 
                                    data=np.asarray(uuu)).rename(var.name)                  # xarray with proper time dimension too
                
            if 'plev' in var.dims:                                                          # If there is also a vertical dimension to work across...
                uuu = [[[tmp[k,i] for i in j] for j in igrid] for k in range(tmp.shape[0])] # Result is a 3D list [p-level,lat,lon]
                res = r_pdef.expand_dims({'plev':var.plev},axis=0).index.copy(deep=True, 
                                    data=np.asarray(uuu)).rename(var.name)                  # xarray with proper pressure-level dimension too                                                
    
        if var.ndim == 3:     # If time and p-level both vary...
            uuu = [[[[tmp[l,k,i] for i in j] for j in igrid] for k in range(tmp.shape[1])] for l in range(tmp.shape[0])] # Result is a 4D list [time,plev,lat,lon]
            res = r_pdef.expand_dims({'plev':var.plev},axis=0).expand_dims({'time':var.time},axis=0).index.copy(deep=True, 
                                data=np.asarray(uuu)).rename(var.name)                      # xarray with time and pressure-level dimensions
                
    return res

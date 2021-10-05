# Precip Project 

## Setup & Run Codes

1. Clone the Repository
```
git clone https://github.com/kpegion/ml-precip.git
cd ml-precip
```

2. Install and activate the environment
```
conda env create -f environment.yml
conda activate ml-precip
python -m ipykernel install --user --name ml-precip --display-name "Python (ml-precip)"
```
3. Launch Jupyter 
All codes are located in `src/`

Main codes:
* `daily_precip_indices.ipynb` - Notebook for producing composites, above/below normal counts, and ML models with *daily* precip indices
* `monthly_precip_indices.ipynb` - Notebook for producing composites, above/below normal counts, and ML models with *monthly* precip indices
* `test_dist.ipynb` - Notebook to calculate and plot histograms of consecutive days with above threshold and below threshold precipitation anomalies; plots fitted distributions to these histograms. 

Data Prep Codes:
* `getDataNASH.ipynb` -- Extracts relevant variables for NASH from ERAI. Takes a long time to run, but only has to run once.
* `calc_NASH.ipynb` -- Calculates amplitude and phase of the NASH from ERAI data for JJA.
* `get_gpcp.sh` -- gets the GPCP precip data from NCEI


### Data 

Years: 1997-2015
This is what I had available and consistent across all indices I was working with. Should be expanded for as long as possible. Precipitation data is the limiting factor with GPCP starting in 1996.

Seasons: Jun-Jul-Aug

1. Global Precipitation Climatology Project (GPCP) Daily Precipitation V1.3, 1x1 deg 
Dates Available: 1996-Present *Limiting factor for analysis-- what can we use for a longer precip dataset?*
Downloaded from: https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/gpcp_v01r03_daily*

2. Monthly Climate Indices 

* Atlantic Multidecadal Oscillation (AMO) 1948-Present
* North Atlantic Oscillation (NAO) 1948-Present
* Pacific Decadal Oscillation (PDO) 1948 Present
* Nino34 1948-Present 

Downloaded from NOAA/ESRL/PSD https://psl.noaa.gov/data/climateindices/list/

Linearly interpolated to daily.

3. RMM Index 
Each day has a RMM Phase and a RMM amplitude

Downloaded from Australian Bureau of Meteorology http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txthttp://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
Dates Available: 1974-Present

4. Pacific North America Weather Regimes

Each day is categorized into one of 5 weather regimes.  using a k-mean cluster analysis.  They are calculated from ERAI Z500 and U250 anomalies following  [Amini and Straus 2019] (https://doi.org/10.1007/s00382-018-4409-7). Identification of weather regimes is done separately for winter (DJF) and summer (JJA). Winter regimes are defined as: Arctic High, Arctic Low, Alaskan Ridge, Pacific Wavetrain,Pacific Trough.  Summer regimes have not been named.  
Figures showing the 5 winter regimes is available here: https://github.com/kpegion/wxregimes/blob/master/figs/ERAI_clusters_5_1980-2015_DJF.png

See https://github.com/kpegion/wxregimes for my codes.

Dates Available: 1979-2015. Can be extended to 2019.

5. Mid-latitude Seasonal Oscillation (MLSO)

Seasonal oscillation in mid-latitudes identified by [Stan and Krishnamurthy 2019](https://link.springer.com/article/10.1007%2Fs00382-019-04827-9).  Calculated using Z500 from ERAI.
Dates Available: 1979-2019

6. North Atlantic Subtropical High (NASH)

Each day is categorized into one of 4 phases of the NASH western ridge and NASH amplitude using ERAI Z850 and U850. Following [Li et al 2012](https://doi.org/10.1007/s00382-011-1214-y) and [Li et al 2011](https://doi.org/10.1175/2010JCLI3829.1), phase is determined by comparing the NASH westrn ridge for a given day to the climatological location of the NASH western ridge and determined to be NW, NE, SW, and SE of the climatological NASH position.  The western ridge is identified for each day after a 5-day running mean is applied by finding the latitude/longitude location where U850=0 line crosses the Z850=1560 gpm line and du/dy>0. Amplitude is determined as the maximum Z850 in the NASH region.  

Dates Available: 1997-2015.  Can be extended to 1979-2019.

### Methods

1. Composites

2. Counts of # of Days in Season above/below normal

3. *Regression Problem*
Standard Linear Regression
Linear Regression with LASSO regularization
Linear Regression with Ridge Regularization
Shallow Neural Network Input(10)->8->8->Output(1)

4. *Classification Problem*
Logisitic Regression
Shallow Neural Network Input(10)->8->8->Output(1)

### Results

#### Diagnostics of Relationship between Common SST and Circulation indices and SEUS Precip Variability






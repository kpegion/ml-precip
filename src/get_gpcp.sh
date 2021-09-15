#!/bin/bash

URL=https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/
OUTPATH=/scratch/kpegion/GPCP
OUTFILE=gpcp_v01r03_daily

# Loop over years
for yr in {1997..2018}
do
   # Get the files for this year and store them in OUTPATH
   wget -r -l1 -nd -nc -A*.nc ${URL}/${yr}/ -P $OUTPATH

   # Concatenate all the files for this year into a single file
   cdo cat $OUTPATH/$OUTFILE_d*_c*.nc $OUTPATH/$OUTFILE.$yr.nc

   # If that is successful, then remove the individual daily files
   if [ $? ]
   then
      rm $OUTPATH/${OUTFILE}_d*_c*.nc
   fi
done


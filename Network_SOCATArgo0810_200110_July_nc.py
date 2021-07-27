#!/usr/bin/env python

# -*- coding: utf-8 -*-

import re
import mod_focus as foc
import FARC_colormaps
import scipy.io
import math
import glob
import datetime,time
from datetime import timedelta
from scipy.signal import *
from scipy.ndimage import filters
from scipy import interpolate
from scipy import stats
import random
import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop

import h5py
import numpy as npy
import matplotlib as mlp
mlp.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
#from pylab import *

mlp.rcParams['font.family']='serif'
mlp.rcParams['font.serif']=['Arial']
mlp.rcParams['font.size']=14
mlp.rcParams['xtick.direction'] = 'out'
mlp.rcParams['ytick.direction'] = 'out'

direct = "/home/biomac2/sommer/Data/NEMO/ORCA025-PIS2DIC_20090101_20091231_5D_CHL.nc"
f = h5py.File(direct, 'r')   # 'r' means that hdf5 file is open in read-only mode
a_group_key1 = f.keys()[2]
a_group_key2 = f.keys()[1]
lon_NEMO = f[a_group_key1]
lat_NEMO = f[a_group_key2]

direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_year_07.nc"
year_list0 = foc.readnc_1d(direct,'year_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_month_07.nc"
month_list0 = foc.readnc_1d(direct,'month_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_day_07.nc"
day_list0 = foc.readnc_1d(direct,'day_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_lat_07.nc"
lat0 = foc.readnc_1d(direct,'lat_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_lon_07.nc"
lon0 = foc.readnc_1d(direct,'lon_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_SSS_07.nc"
SSS_list0 = foc.readnc_1d(direct,'SSS_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_SST_07.nc"
SST_list0 = foc.readnc_1d(direct,'SST_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_SSH_07.nc"
SSH_list0 = foc.readnc_1d(direct,'SSH_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_CHL1_07.nc"
CHL1_list0 = foc.readnc_1d(direct,'CHL1_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_CHL2_07.nc"
CHL2_list0 = foc.readnc_1d(direct,'CHL2_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_MLD_07.nc"
MLD_list0 = foc.readnc_1d(direct,'MLD_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_pCO2_atm_07.nc"
pCO2_atm_list0 = foc.readnc_1d(direct,'pCO2_atm_list')
direct = "/home/biomac2/sommer/Data/SOCAT_NEMO_train2_nc/Data_training_pCO2_07.nc"
pCO2_list0 = foc.readnc_1d(direct,'pCO2_list')

direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_year_2010_07.nc"
year_list1 = foc.readnc_1d(direct,'year_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_month_2010_07.nc"
month_list1 = foc.readnc_1d(direct,'month_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_day_2010_07.nc"
day_list1 = foc.readnc_1d(direct,'day_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_lat_2010_07.nc"
lat1 = foc.readnc_1d(direct,'lat_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_lon_2010_07.nc"
lon1 = foc.readnc_1d(direct,'lon_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SSS_2010_07.nc"
SSS_list1 = foc.readnc_1d(direct,'SSS_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SST_2010_07.nc"
SST_list1 = foc.readnc_1d(direct,'SST_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SSH_2010_07.nc"
SSH_list1 = foc.readnc_1d(direct,'SSH_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_CHL1_2010_07.nc"
CHL1_list1 = foc.readnc_1d(direct,'CHL1_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_CHL2_2010_07.nc"
CHL2_list1 = foc.readnc_1d(direct,'CHL2_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_MLD_2010_07.nc"
MLD_list1 = foc.readnc_1d(direct,'MLD_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_pCO2_atm_2010_07.nc"
pCO2_atm_list1 = foc.readnc_1d(direct,'pCO2_atm_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_pCO2_2010_07.nc"
pCO2_list1 = foc.readnc_1d(direct,'pCO2_list')

direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_year_2009_07.nc"
year_list2 = foc.readnc_1d(direct,'year_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_month_2009_07.nc"
month_list2 = foc.readnc_1d(direct,'month_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_day_2009_07.nc"
day_list2 = foc.readnc_1d(direct,'day_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_lat_2009_07.nc"
lat2 = foc.readnc_1d(direct,'lat_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_lon_2009_07.nc"
lon2 = foc.readnc_1d(direct,'lon_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SSS_2009_07.nc"
SSS_list2 = foc.readnc_1d(direct,'SSS_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SST_2009_07.nc"
SST_list2 = foc.readnc_1d(direct,'SST_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SSH_2009_07.nc"
SSH_list2 = foc.readnc_1d(direct,'SSH_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_CHL1_2009_07.nc"
CHL1_list2 = foc.readnc_1d(direct,'CHL1_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_CHL2_2009_07.nc"
CHL2_list2 = foc.readnc_1d(direct,'CHL2_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_MLD_2009_07.nc"
MLD_list2 = foc.readnc_1d(direct,'MLD_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_pCO2_atm_2009_07.nc"
pCO2_atm_list2 = foc.readnc_1d(direct,'pCO2_atm_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_pCO2_2009_07.nc"
pCO2_list2 = foc.readnc_1d(direct,'pCO2_list')

direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_year_2008_07.nc"
year_list3 = foc.readnc_1d(direct,'year_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_month_2008_07.nc"
month_list3 = foc.readnc_1d(direct,'month_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_day_2008_07.nc"
day_list3 = foc.readnc_1d(direct,'day_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_lat_2008_07.nc"
lat3 = foc.readnc_1d(direct,'lat_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_lon_2008_07.nc"
lon3 = foc.readnc_1d(direct,'lon_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SSS_2008_07.nc"
SSS_list3 = foc.readnc_1d(direct,'SSS_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SST_2008_07.nc"
SST_list3 = foc.readnc_1d(direct,'SST_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_SSH_2008_07.nc"
SSH_list3 = foc.readnc_1d(direct,'SSH_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_CHL1_2008_07.nc"
CHL1_list3 = foc.readnc_1d(direct,'CHL1_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_CHL2_2008_07.nc"
CHL2_list3 = foc.readnc_1d(direct,'CHL2_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_MLD_2008_07.nc"
MLD_list3 = foc.readnc_1d(direct,'MLD_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_pCO2_atm_2008_07.nc"
pCO2_atm_list3 = foc.readnc_1d(direct,'pCO2_atm_list')
direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_SOCAT_20082010_nc/Data_training_pCO2_2008_07.nc"
pCO2_list3 = foc.readnc_1d(direct,'pCO2_list')

year_list = npy.concatenate((year_list0,year_list1,year_list2,year_list3),axis =0)
month_list = npy.concatenate((month_list0,month_list1,month_list2,month_list3),axis =0)
day_list = npy.concatenate((day_list0,day_list1,day_list2,day_list3),axis =0)
lat = npy.concatenate((lat0,lat1,lat2,lat3),axis =0)
lon = npy.concatenate((lon0,lon1,lon2,lon3),axis =0)
SST_list = npy.concatenate((SST_list0,SST_list1,SST_list2,SST_list3),axis =0)
SSS_list = npy.concatenate((SSS_list0,SSS_list1,SSS_list2,SSS_list3),axis =0)
SSH_list = npy.concatenate((SSH_list0,SSH_list1,SSH_list2,SSH_list3),axis =0)
CHL1_list = npy.concatenate((CHL1_list0,CHL1_list1,CHL1_list2,CHL1_list3),axis =0)
CHL2_list = npy.concatenate((CHL2_list0,CHL2_list1,CHL2_list2,CHL2_list3),axis =0)
MLD_list = npy.concatenate((MLD_list0,MLD_list1,MLD_list2,MLD_list3),axis =0)
pCO2_atm_list = npy.concatenate((pCO2_atm_list0,pCO2_atm_list1,pCO2_atm_list2,pCO2_atm_list3),axis =0)
pCO2_list = npy.concatenate((pCO2_list0,pCO2_list1,pCO2_list2,pCO2_list3),axis =0)

index1 = npy.zeros(10000000)
p = 0
for i in npy.arange(0,len(pCO2_list),1):
    if pCO2_list[i] == 0. or npy.abs(pCO2_list[i]) > 1000.:
       index1[p] = i
       p = p + 1
index1 = index1[:p]
year_list = npy.delete(year_list, index1)
month_list = npy.delete(month_list, index1)
day_list = npy.delete(day_list, index1)
lat = npy.delete(lat, index1)
lon = npy.delete(lon, index1)
SST_list = npy.delete(SST_list, index1)
SSS_list = npy.delete(SSS_list, index1)
SSH_list = npy.delete(SSH_list, index1)
CHL1_list = npy.delete(CHL1_list, index1)
CHL2_list = npy.delete(CHL2_list, index1)
MLD_list = npy.delete(MLD_list, index1)
pCO2_atm_list = npy.delete(pCO2_atm_list, index1)
pCO2_list = npy.delete(pCO2_list, index1)

print month_list
print pCO2_list.shape

CHL_list = CHL1_list + CHL2_list

SSS_anom = SSS_list - npy.nanmean(SSS_list)
SST_anom = SST_list - npy.nanmean(SST_list)
SSH_anom = SSH_list - npy.nanmean(SSH_list)
CHL_anom = CHL_list - npy.nanmean(CHL_list)
MLD_anom = MLD_list - npy.nanmean(MLD_list)
pCO2_atm_anom = pCO2_atm_list - npy.nanmean(pCO2_atm_list)

CHL_log = npy.zeros(len(CHL_list))
MLD_log = npy.zeros(len(MLD_list))

for t in npy.arange(0,len(CHL_list),1):
    CHL_log[t] = npy.log10(CHL_list[t])
    MLD_log[t] = npy.log10(MLD_list[t])

SSS_region = (SSS_list - npy.nanmean(SSS_list))/npy.nanstd(SSS_list)
SST_region = (SST_list - npy.nanmean(SST_list))/npy.nanstd(SST_list)
SSH_region =(SSH_list - npy.nanmean(SSH_list))/npy.nanstd(SSH_list)
CHL_region = (CHL_log)/npy.nanstd(CHL_log)
pCO2_region = (pCO2_list - npy.nanmean(pCO2_list))/npy.nanstd(pCO2_list)
MLD_region = (MLD_log - npy.nanmean(MLD_log))/npy.nanstd(MLD_log)
pCO2_atm_region = (pCO2_atm_list - npy.nanmean(pCO2_atm_list))/npy.nanstd(pCO2_atm_list)

SSS_anom_region = (SSS_anom - npy.nanmean(SSS_anom))/npy.nanstd(SSS_anom)
SST_anom_region = (SST_anom - npy.nanmean(SST_anom))/npy.nanstd(SST_anom)
SSH_anom_region = (SSH_anom - npy.nanmean(SSH_anom))/npy.nanstd(SSH_anom)
CHL_anom_region = (CHL_anom)/npy.nanstd(CHL_anom)
MLD_anom_region = (MLD_anom - npy.nanmean(MLD_anom))/npy.nanstd(MLD_anom)
pCO2_atm_anom_region = (pCO2_atm_anom - npy.nanmean(pCO2_atm_anom))/npy.nanstd(pCO2_atm_anom)

lat_list1 = npy.zeros(len(lat))
lon_list1 = npy.zeros(len(lat))
lon_list2 = npy.zeros(len(lat))

for i in npy.arange(0,len(lat),1):
    lat_list1[i] = npy.sin(lat[i]*npy.pi/180.)
    lon_list1[i] = npy.cos(lon[i]*npy.pi/180.)
    lon_list2[i] = npy.sin(lon[i]*npy.pi/180.)

data_predictors = npy.column_stack((SSS_region,SST_region,SSH_region,CHL_region,MLD_region,pCO2_atm_region,lat_list1,lon_list1,lon_list2,SSS_anom_region,SST_anom_region,SSH_anom_region,CHL_anom_region,MLD_anom_region,pCO2_atm_anom_region))

pCO2_SOCAT_list1_tot = npy.zeros(100000)
pCO2_reconstr_list1_tot = npy.zeros(100000)
pCO2_reconstr_anom_list1_tot = npy.zeros(100000)

pCO2_train_val_extr_tot = npy.zeros(100000)
pCO2_train_anom_val_extr_tot = npy.zeros(100000)
pCO2_percip_val_extr_tot = npy.zeros(100000)
pCO2_percip_anom_val_extr_tot = npy.zeros(100000)

j_Val_tot = 0
j_SOCAT_tot = 0

for numb_of_model in npy.arange(0,4,1):
    index1 = npy.zeros(len(pCO2_region), dtype=npy.int)
    p = 0
    for i in npy.arange(numb_of_model,len(pCO2_region),4):
        index1[p] = i
        p = p + 1
    index1 = index1[:p]
    data_train1 = npy.delete(data_predictors, index1, axis = 0)
    pCO2_list_train1 = npy.delete(pCO2_region, index1)
    lon_list_degree1 = npy.delete(lon, index1)
    lat_list_degree1 = npy.delete(lat, index1)
    year_list1 = npy.delete(year_list, index1)
    month_list1 = npy.delete(month_list, index1)
    day_list1 = npy.delete(day_list, index1)
    data_eval = npy.zeros((len(index1),15))
    pCO2_list_eval = npy.zeros(len(index1))
    for i in npy.arange(0,len(index1),1):
        data_eval[i,:] = data_predictors[index1[i],:]
        pCO2_list_eval[i] = pCO2_region[index1[i]]
    index2 = npy.zeros(len(pCO2_list_train1), dtype=npy.int)
    p = 0
    for i in npy.arange(numb_of_model,len(pCO2_list_train1),3):
        index2[p] = i
        p = p + 1
    index2 = index2[:p]
    data_train = npy.delete(data_train1, index2, axis = 0)
    pCO2_list_train = npy.delete(pCO2_list_train1, index2)
    year_train = npy.delete(year_list1, index2)
    month_train = npy.delete(month_list1, index2)
    day_train = npy.delete(day_list1, index2)
    lon_train = npy.delete(lon_list_degree1, index2)
    lat_train = npy.delete(lat_list_degree1, index2)

    data_val = npy.zeros((len(index2),15))
    pCO2_list_val = npy.zeros(len(index2))
    pCO2_list_val_tot = npy.zeros(len(index2))
    lon_degree_val = npy.zeros(len(index2))
    lat_degree_val = npy.zeros(len(index2))
    year_list_val = npy.zeros(len(index2))
    month_list_val = npy.zeros(len(index2))
    day_list_val = npy.zeros(len(index2))
    for i in npy.arange(0,len(index2),1):
        data_val[i,:] = data_train1[index2[i],:]
        pCO2_list_val[i] = pCO2_list_train1[index2[i]]
        lon_degree_val[i] = lon_list_degree1[index2[i]]
        lat_degree_val[i] = lat_list_degree1[index2[i]]
        year_list_val[i] = year_list1[index2[i]]
        month_list_val[i] = month_list1[index2[i]]
        day_list_val[i] = day_list1[index2[i]]

    print 'Version', numb_of_model
    print 'Training data!!!!!!!!!!!!!!!!!!!!!!!!!!!', data_train.shape, pCO2_list_train.shape
    print 'Validation data!!!!!!!!!!!!!!!!!!!!!!!!!', data_val.shape, pCO2_list_val.shape
    print 'Evaluation data!!!!!!!!!!!!!!!!!!!!!!!!!', data_eval.shape, pCO2_list_eval.shape

    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    model = Sequential()
    model.add(Dense(20, input_dim=15, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(25, init='glorot_uniform'))
    model.add(Activation('tanh'))
    #model.add(Dense(60, init='glorot_uniform'))
    #model.add(Activation('tanh'))
    #model.add(Dense(30, input_dim=60, init='glorot_uniform'))
    #model.add(Activation('tanh'))
    #model.add(Dense(25, input_dim=20, init='glorot_uniform'))
    #model.add(Activation('tanh'))
    #model.add(Dense(15, input_dim=25, init='glorot_uniform'))
    #model.add(Activation('tanh'))
    model.add(Dense(10, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(1, init='glorot_uniform'))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    L = model.compile(loss='mse',optimizer=rmsprop)
    print model.get_weights()
    H = model.fit(data_train, pCO2_list_train,nb_epoch=1600,callbacks=[earlyStopping],batch_size=20,validation_data=(data_eval,pCO2_list_eval))
    score = model.evaluate(data_val, pCO2_list_val, batch_size=20)
    print model.get_weights()
    print model.summary()
    print model.get_config()

    preds1 = model.predict(data_train)
    preds_list = preds1[:,0]
    pCO2_matrix = preds_list * npy.nanstd(pCO2_list) + npy.nanmean(pCO2_list)
    pCO2_list_test = pCO2_list_train * npy.nanstd(pCO2_list) + npy.nanmean(pCO2_list)

    print 'TEST!!!!!!!!!!!!!!!!!!!'
    print 'TOTAL'
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print'Validation'
    print 'Personr',scipy.stats.pearsonr(pCO2_matrix,pCO2_list_test)
    print 'RMS', npy.sqrt(npy.nanmean((pCO2_matrix-pCO2_list_test)**2))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pCO2_matrix,pCO2_list_test)
    print 'R2', r_value**2
    print 'Bias', npy.nanmean(pCO2_matrix) - npy.nanmean(pCO2_list_test)

    fo = open("/home/users/asommer/Cost_Function_FFN_Network_Design_SOCATArgo_20012010/Cost_Function_SOCATArgo0810_20012010_July_test1_" + str(numb_of_model) + ".txt","wb")
    for i in npy.arange(0,len(H.history['loss']),1):
        fo.write(str(i)+';'+str(npy.log10(H.history['loss'][i]))+';'+str(npy.log10(H.history['val_loss'][i]))+ "\n")
    fo.close()

    pCO2_reconstr = npy.zeros(len(lat_degree_val))
    pCO2_origin = npy.zeros(len(lat_degree_val))

    k = 0
    SSS_list_predict1 = npy.zeros(len(lat_train))
    SST_list_predict1 = npy.zeros(len(lat_train))
    SSH_list_predict1 = npy.zeros(len(lat_train))
    CHL_log_predict1 = npy.zeros(len(lat_train))
    MLD_log_predict1 = npy.zeros(len(lat_train))
    pCO2_atm_list_predict1 = npy.zeros(len(lat_train))
    lat_predict_list11 = npy.zeros(len(lat_train))
    lon_predict_list11 = npy.zeros(len(lat_train))
    lon_predict_list21 = npy.zeros(len(lat_train))
    SSS_anom_predict1 = npy.zeros(len(lat_train))
    SST_anom_predict1 = npy.zeros(len(lat_train))
    SSH_anom_predict1 = npy.zeros(len(lat_train))
    CHL_anom_log_predict1 = npy.zeros(len(lat_train))
    MLD_anom_log_predict1 = npy.zeros(len(lat_train))
    pCO2_atm_anom_predict1 = npy.zeros(len(lat_train))

    for time_year in npy.arange(2008,2011,1):
        for num_file in npy.arange(36,42,1):
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_year_"+str(time_year)+"_"+str(num_file)+".nc"
            year_predict = foc.readnc_1d(direct,'year_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_month_"+str(time_year)+"_"+str(num_file)+".nc"
            month_predict = foc.readnc_1d(direct,'month_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_day_"+str(time_year)+"_"+str(num_file)+".nc"
            day_predict = foc.readnc_1d(direct,'day_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_lat_"+str(time_year)+"_"+str(num_file)+".nc"
            lat_predict = foc.readnc_1d(direct,'lat_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_lon_"+str(time_year)+"_"+str(num_file)+".nc"
            lon_predict = foc.readnc_1d(direct,'lon_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_SSS_"+str(time_year)+"_"+str(num_file)+".nc"
            SSS_list_predict = foc.readnc_1d(direct,'SSS_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_SST_"+str(time_year)+"_"+str(num_file)+".nc"
            SST_list_predict = foc.readnc_1d(direct,'SST_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_SSH_"+str(time_year)+"_"+str(num_file)+".nc"
            SSH_list_predict = foc.readnc_1d(direct,'SSH_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_CHL1_"+str(time_year)+"_"+str(num_file)+".nc"
            CHL1_list_predict = foc.readnc_1d(direct,'CHL1_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_CHL2_"+str(time_year)+"_"+str(num_file)+".nc"
            CHL2_list_predict = foc.readnc_1d(direct,'CHL2_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_MLD_"+str(time_year)+"_"+str(num_file)+".nc"
            MLD_list_predict = foc.readnc_1d(direct,'MLD_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_pCO2_atm_"+str(time_year)+"_"+str(num_file)+".nc"
            pCO2_atm_list_predict = foc.readnc_1d(direct,'pCO2_atm_list')
            direct = "/home/biomac2/sommer/Data/Argo_NEMO_FFNN_Atl_20082010_nc/Data_predict_pCO2_"+str(time_year)+"_"+str(num_file)+".nc"
            dpCO2_list_predict = foc.readnc_1d(direct,'pCO2_list')

            CHL_list_predict = CHL1_list_predict + CHL2_list_predict

            SSS_anom_predict = SSS_list_predict - npy.nanmean(SSS_list)
            SST_anom_predict = SST_list_predict - npy.nanmean(SST_list)
            SSH_anom_predict = SSH_list_predict - npy.nanmean(SSH_list)
            CHL_anom_predict = CHL_list_predict - npy.nanmean(CHL_list)
            MLD_anom_predict = MLD_list_predict - npy.nanmean(MLD_list)
            pCO2_atm_anom_predict = pCO2_atm_list_predict - npy.nanmean(pCO2_atm_list)

            CHL_log_predict = npy.zeros(len(CHL_list_predict))
            MLD_log_predict = npy.zeros(len(MLD_list_predict))

            for t in npy.arange(0,len(CHL_list_predict),1):
                CHL_log_predict[t] = npy.log10(CHL_list_predict[t])
                MLD_log_predict[t] = npy.log10(MLD_list_predict[t])

            SSS_list_predict = (SSS_list_predict - npy.nanmean(SSS_list))/npy.nanstd(SSS_list)
            SST_list_predict = (SST_list_predict - npy.nanmean(SST_list))/npy.nanstd(SST_list)
            SSH_list_predict = (SSH_list_predict - npy.nanmean(SSH_list))/npy.nanstd(SSH_list)
            CHL_list_predict = (CHL_log_predict)/npy.nanstd(CHL_log)
            MLD_list_predict = (MLD_log_predict - npy.nanmean(MLD_log))/npy.nanstd(MLD_log)
            pCO2_atm_list_predict = (pCO2_atm_list_predict - npy.nanmean(pCO2_atm_list))/npy.nanstd(pCO2_atm_list)

            SSS_anom_predict = (SSS_anom_predict - npy.nanmean(SSS_anom))/npy.nanstd(SSS_anom)
            SST_anom_predict = (SST_anom_predict - npy.nanmean(SST_anom))/npy.nanstd(SST_anom)
            SSH_anom_predict = (SSH_anom_predict - npy.nanmean(SSH_anom))/npy.nanstd(SSH_anom)
            pCO2_atm_anom_predict = (pCO2_atm_anom_predict - npy.nanmean(pCO2_atm_anom))/npy.nanstd(pCO2_atm_anom)
            CHL_anom_log_predict = (CHL_anom_predict)/npy.nanstd(CHL_anom)
            MLD_anom_log_predict = (MLD_anom_predict - npy.nanmean(MLD_anom))/npy.nanstd(MLD_anom)

            lat_predict_list1 = npy.zeros(len(lat_predict))
            lon_predict_list1 = npy.zeros(len(lat_predict))
            lon_predict_list2 = npy.zeros(len(lat_predict))

            for i in npy.arange(0,len(lat_predict),1):
                lat_predict_list1[i] = npy.sin(lat_predict[i]*npy.pi/180.)
                lon_predict_list1[i] = npy.cos(lon_predict[i]*npy.pi/180.)
                lon_predict_list2[i] = npy.sin(lon_predict[i]*npy.pi/180.)

            data_reconstr = npy.column_stack((SSS_list_predict,SST_list_predict,SSH_list_predict,CHL_list_predict,MLD_list_predict,pCO2_atm_list_predict,lat_predict_list1,lon_predict_list1,lon_predict_list2,SSS_anom_predict,SST_anom_predict,SSH_anom_predict,CHL_anom_log_predict,MLD_anom_log_predict,pCO2_atm_anom_predict))

            pCO2_matrix = npy.zeros(len(lat_predict))
            preds1 = model.predict(data_reconstr)
            preds_list = preds1[:,0]
            pCO2_matrix = preds_list * npy.nanstd(pCO2_list) + npy.nanmean(pCO2_list)

            if time_year == 2008 or time_year == 2009 or time_year == 2010:
               for j in npy.arange(0,len(lat_degree_val),1):
                   for i in npy.arange(0,len(lat_predict),1):
                       if year_list_val[j] == year_predict[i] and month_list_val[j] == month_predict[i] and day_list_val[j] == day_predict[i] and npy.abs(lat_degree_val[j]- lat_predict[i]) < 0.001 and npy.abs(lon_degree_val[j] - lon_predict[i]) < 0.001:
                          pCO2_reconstr[k] = pCO2_matrix[i]
                          pCO2_origin[k] = pCO2_list_val[j] * npy.nanstd(pCO2_list) + npy.nanmean(pCO2_list)
                          k = k + 1
                          break

            time = npy.arange(0,len(pCO2_matrix),1) 
            foc.write_list('/home/biomac2/sommer/Data/Network_Design_SOCATArgo_20012010_nc/pCO2_Argo0810_reconstr_July_'+str(time_year)+str(num_file)+'_test1_'+str(numb_of_model)+'.nc',time,pCO2_matrix,'pCO2')
            foc.write_list('/home/biomac2/sommer/Data/Network_Design_SOCATArgo_20012010_nc/Lat_Argo0810_reconstr_July_'+str(time_year)+str(num_file)+'_test1_'+str(numb_of_model)+'.nc',time,lat_predict,'lat')
            foc.write_list('/home/biomac2/sommer/Data/Network_Design_SOCATArgo_20012010_nc/Lon_Argo0810_reconstr_July_'+str(time_year)+str(num_file)+'_test1_'+str(numb_of_model)+'.nc',time,lon_predict,'lon')

    pCO2_reconstr = pCO2_reconstr[:k]
    pCO2_origin = pCO2_origin[:k]

    print pCO2_reconstr, pCO2_origin
    print 'TOTAL'
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print'Validation'
    print 'Personr',scipy.stats.pearsonr(pCO2_origin,pCO2_reconstr)
    print 'RMS', npy.sqrt(npy.nanmean((pCO2_origin-pCO2_reconstr)**2))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pCO2_origin,pCO2_reconstr)
    print 'R2', r_value**2
    print 'Bias', npy.nanmean(pCO2_origin) - npy.nanmean(pCO2_reconstr)
    print 'Bias', npy.nanmean(npy.abs(pCO2_origin) - npy.abs(pCO2_reconstr))
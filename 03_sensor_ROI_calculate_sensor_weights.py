# Import libraries
import os
import sys
import time
import mne
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import multiprocessing as mp
import warnings



def spectralevents_ts2tfr (S,fVec,Fs,width):
		# spectralevents_ts2tfr(S,fVec,Fs,width);
		#
		# Calculates the TFR (in spectral power) of a time-series waveform by 
		# convolving in the time-domain with a Morlet wavelet.                            
		#
		# Input
		# -----
		# S    : signals = time x Trials      
		# fVec    : frequencies over which to calculate TF energy        
		# Fs   : sampling frequency
		# width: number of cycles in wavelet (> 5 advisable)  
		#
		# Output
		# ------
		# t    : time
		# f    : frequency
		# B    : phase-locking factor = frequency x time
		#     
		# Adapted from Ole Jensen's traces2TFR in the 4Dtools toolbox.


		S = S.T
		numTrials = S.shape[0]
		numSamples = S.shape[1]
		numFrequencies = len(fVec)

		tVec = np.arange(numSamples)/Fs

		TFR = []
		# Trial Loop
		for i in np.arange(numTrials):
			B = np.zeros((numFrequencies, numSamples))
			# Frequency loop
			for j in np.arange(numFrequencies):
				B[j,:] = energyvec(fVec[j], signal.detrend(S[i,:]), Fs, width)
			TFR.append(B)

		TFR = np.asarray(TFR)

		return TFR, tVec, fVec


def energyvec(f,s,Fs,width):
	# Return a vector containing the energy as a
	# function of time for frequency f. The energy
	# is calculated using Morlet's wavelets. 
	# s : signal
	# Fs: sampling frequency
	# width : width of Morlet wavelet (>= 5 suggested).

	dt = 1/Fs
	sf = float(f)/float(width)
	st = 1/(2 * np.pi * sf)

	t= np.arange(-3.5*st, 3.5*st, dt)
	m = morlet(f, t, width)

	y = np.convolve(s, m)
	y = 2 * (dt * np.abs(y))**2
	lowerLimit = int(np.ceil(len(m)/2))
	upperLimit = int(len(y)-np.floor(len(m)/2)+1) #lowerLimit + size)
	y = y[lowerLimit:upperLimit]

	return y

def morlet(f,t,width):
	# Morlet's wavelet for frequency f and time t. 
	# The wavelet will be normalized so the total energy is 1.
	# width defines the ``width'' of the wavelet. 
	# A value >= 5 is suggested.
	#
	# Ref: Tallon-Baudry et al., J. Neurosci. 15, 722-734 (1997)

	sf = f/width
	st = 1/(2 * np.pi * sf)
	A = 1/(st * np.sqrt(2 * np.pi))
	y = A * np.exp(-t**2 / (2 * st**2)) * np.exp(1j * 2 * np.pi * f * t)

	return y


def get_power(TFR, tmin, tmax, fmin, fmax):
	# get TFR power averaged over time and frequency
	# TFR = TFR object with baseline correction and averaged across epochs
	# TFR.data.shape should be (channel, frequency, time) = (1, 10, 3401)

	#convert TFR object to raw data array and collapse to 2d array [freq x time]
	TFR_array = np.mean(TFR.data, axis=0) 

	#convert to dataframe
	df_TFR = pd.DataFrame(data=TFR_array, index=TFR.freqs, columns=TFR.times)

	#truncate the dataframe to time and freq range of interest
	df_TFR = df_TFR.truncate(before=fmin, after=fmax)
	df_TFR = df_TFR.truncate(before=tmin, after=tmax, axis="columns")

	#calc average power
	P = abs(df_TFR.mean().mean())

	return P




if __name__ == "__main__":


	# Find subjects to be analysed and take only subjects with more than 55 epochs
	homeDir = os.path.expanduser("~")
	dataDir = homeDir + '/camcan/'
	camcanCSV = dataDir + 'proc_data/oneCSVToRuleThemAll.csv'
	subjectData = pd.read_csv(camcanCSV)
	subjectData = subjectData[subjectData['numEpochs'] > 55]
	subjectIDs = subjectData['SubjectID'].tolist() 

	############ HACK FOR TESTING ##############################################
	subjectIDs = subjectIDs[0:3]
	############################################################################

	# ROI channels
	ROI_L = ['MEG0221', 'MEG0321', 'MEG0341', 'MEG0211', 'MEG0411', 'MEG0231', 'MEG0441']
	ROI_R = ['MEG1311', 'MEG1231', 'MEG1221', 'MEG1321', 'MEG1121', 'MEG1341', 'MEG1131']
	ROI_sides = [ROI_L, ROI_R]

	# integer number of cycles in wavelet (> 5 advisable)
	width = 10

	# time ranges (s)
	epoch_tmin = -1.7
	epoch_tmax = 1.7
	baseline_tmin = -1.0
	baseline_tmax = -0.5
	movement_tmin = -0.1
	movement_tmax = 0.5
	postmovement_tmin = 0.5
	postmovement_tmax = 1.25
	MRGB_tmin = -0.2
	MRGB_tmax = 0.2

	# beta range (Hz)
	beta_fmin = 15
	beta_fmax = 30

	# gamma range (Hz)
	gamma_fmin = 60
	gamma_fmax = 90


	# vector of frequencies in TFR (measured in integer number of Hz)
	fmin = 8		
	fmax = 36			
	fstep = 1 			
	fVec = np.arange(fmin, fmax+fstep, fstep)

	df_weights = pd.DataFrame(columns=['channel', 'location', 'weight'])

	side = 0

	for region in ROI_sides:

		side=side+1
		if side == 1:
			loc = 'left'
		if side == 2:
			loc = 'right'

		ROI = region


		TFR_ROI = []

		for subj in subjectIDs:

			print('beta_and_mu' + '_' + subj + '_' + loc)

			df_w = pd.DataFrame(columns=['subject', 'channel', 'location', 'weight', 'phi', 'tau', 'chi', 'phi_numerator', 'tau_numerator', 'chi_numerator'])
			
			TFRs_ROI = []
			P_Movement_ROI = []
			P_PostMovement_ROI = []

			# set epoch directory  
			homeDir = os.path.expanduser("~")      
			dataDir = homeDir + '/camcan/proc_data/TaskSensorAnalysis_transdef'
			epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
			epochFif = os.path.join(dataDir, subj, epochFifFilename)

			if os.path.exists(epochFif):	

				for chn in ROI:

					# read the epochs
					epochs = mne.read_epochs(epochFif)
					Fs = epochs.info['sfreq']

					# Extract the data
					epochs.pick_channels([chn])
					epochData = np.squeeze(epochs.get_data())
					
					#Make a TFR for each epoch [trials x frequency x time]
					TFR, tVec, fVec = spectralevents_ts2tfr(epochData.T, fVec, Fs, width)

					#create TFR object
					TFR = np.expand_dims(TFR, axis=1) #add extra dimension to represent channel
					info = mne.create_info(ch_names=[chn], sfreq=Fs, ch_types='mag')
					times = np.arange(epoch_tmin, epoch_tmax+1/Fs, 1/Fs)  
					freqs = np.arange(fmin, fmax+fstep, fstep)  
					TFR = mne.time_frequency.EpochsTFR(info=info, data=TFR, times=times, freqs=freqs)

					#average TFR object over all trials and then apply baseline correction
					TFR_ave = TFR.average()
					TFR_ave.apply_baseline(baseline=(baseline_tmin, baseline_tmax), mode='logratio')

					#append TFR object to list of all TFRs in ROI
					TFRs_ROI.append(TFR_ave)

					#calculate avg power over ERD and PMBR and append to list 
					P_Movement = get_power(TFR_ave, movement_tmin, movement_tmax, beta_fmin, beta_fmax)
					P_PostMovement =  get_power(TFR_ave, postmovement_tmin, postmovement_tmax, beta_fmin, beta_fmax)
					P_Movement_ROI.append(P_Movement)
					P_PostMovement_ROI.append(P_PostMovement)


			# calculate channel weighting for ROI
			P_M = P_Movement_ROI
			P_PM = P_PostMovement_ROI
			chi = sum(P_M) / sum([x+y for x,y in zip(P_M, P_PM)] )
			w = []
			taus = []
			phis = []
			taus_num = []
			phis_num = []			
			for i in range(len(ROI)):
				tau = P_PM[i]/sum(P_PM)
				phi = P_M[i]/sum(P_M)
				weight = chi*phi + (1-chi)*tau
				w.append(weight)
				taus.append(tau)
				phis.append(phi)
				taus_num.append(P_PM[i])
				phis_num.append(P_M[i])				

			#plot weights statistics
			df_w['subject'] = [subj]*len(ROI)
			df_w['channel'] = ROI_L 
			df_w['location'] = loc
			df_w['weight']= w
			df_w['phi']= phis
			df_w['tau']= taus		
			df_w['chi']= chi
			df_w['phi_numerator']= phis_num
			df_w['tau_numerator']= taus_num		
			df_w['chi_numerator']= sum(P_M)			
			
			
			df_weights = df_weights.append(df_w)

		
	# write weight parameters to csv file
	homeDir = os.path.expanduser("~")
	outDir = homeDir + '/camcan/' 
	fname = outDir + 'weights.csv'
	df_weights.to_csv(fname, index=False)
	
	




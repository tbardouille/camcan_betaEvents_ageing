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
		#
		# See also SPECTRALEVENTS, SPECTRALEVENTS_FIND, SPECTRALEVENTS_VIS.

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

	# vector of frequencies in TFR (measured in integer number of Hz)
	fmin = 5			
	fmax = 105			
	fstep = 1 			
	fVec = np.arange(fmin, fmax+fstep, fstep)

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


	# load in channel weights
	homeDir = os.path.expanduser("~")   
	dataDir = homeDir + '/camcan/'
	weight_stats = 'weights.csv'
	csvFile = os.path.join(dataDir, weight_stats)
	df_w = pd.read_csv(csvFile)


	for subj in subjectIDs:


		# set epoch directory  
		homeDir = os.path.expanduser("~")      
		dataDir = homeDir + '/camcan/proc_data/TaskSensorAnalysis_transdef'
		epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
		epochFif = os.path.join(dataDir, subj, epochFifFilename)

		for chns in ROI_sides: # i.e. left or right

			#determine which side we are on based on one channel in the ROI
			if chns[0] == 'MEG0221':
				side = 'left'
			if chns[0] == 'MEG1311':
				side = 'right'
				

			# keep only weights for this subject and side
			df_w_subj = df_w.loc[df_w['subject'] == subj]
			df_w_subj_side = df_w_subj.loc[df_w_subj['location'] == side]

			# if the side is right, we must change dataframe to represent that (because weight_stats.csv is written with only left side channels)
			if side == 'right':
				chn_names = dict(zip(ROI_L, ROI_R))			
				df_w_subj_side = df_w_subj_side.replace({'channel':chn_names})

			# make a dictionary for channel: weight
			chn_weights = dict(zip(df_w_subj_side.channel, df_w_subj_side.weight))

			ROI_TFRs = []

			for chn in chns:

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

				# Apply weighting to the TFR data
				TFR.data = chn_weights[chn]*TFR.data

				# re-arrange TFR array index so we fool the computer to think 63 channels and 1 epoch i.e. (63, 1, 29, 3401) to (1, 63, 29, 3401)
				TFR.data = np.moveaxis(TFR.data, 1, 0)

				# Append to list of ROI_TFRs
				ROI_TFRs.append(TFR.data)


			# join all ROI TFRs data to make one array of shape (7, 63, 29, 3401)
			weighted_TFR = np.concatenate(ROI_TFRs, axis=0)

			# change back into TFR object (note that channels and epochs are switched)
			info = mne.create_info(ch_names= weighted_TFR.shape[1], sfreq=Fs, ch_types='mag')
			TFR = mne.time_frequency.EpochsTFR(info=info, data=weighted_TFR, times=times, freqs=freqs)

			# average across channels (using the average across epochs function)
			ROI_TFR = TFR.average() #(63, 29, 3401)

			#save a TFR object for each epoch
			epochs = ROI_TFR.info.get('ch_names')
			
			for epoch in epochs:
				ROI_TFR_copy = ROI_TFR.copy()

				#set ouput filename
				homeDir = os.path.expanduser("~")   
				dataDir = homeDir + '/camcan/'

				outDir = homeDir + '/camcan/'
				fname = outDir  + str(subj) + '_' + side + '_' + 'trial=' + epoch + '_' + 'frange=' + str(fmin) + '-' + str(fmax) + 'Hz_fstep=' + str(fstep) + 'Hz-tfr.h5'
				if not os.path.exists(outDir):
					os.makedirs(outDir)

				epoch = str(epoch)	

				TFR = ROI_TFR_copy.pick_channels(ch_names = [epoch])
				TFR.save(fname, overwrite=True)




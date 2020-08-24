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
#import support_functions.spectralevents_functions_brendanEdgeEffectsEdits as tse
import logging
import multiprocessing as mp
import warnings
from scipy import stats

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



def percent_pixels(spectrogram_thresholded, span):

	percents = []
	for timepoint in spectrogram_thresholded.T:
		count = np.count_nonzero(timepoint == 1)
		percent = count / span
		percents.append(percent)
		percent_of_pixels = np.asarray(percents) 

	return percent_of_pixels


if __name__ == "__main__":

	# Find subjects to be analysed and take only subjects with more than 55 epochs
	homeDir = os.path.expanduser("~")
	dataDir = homeDir + '/camcan/'
	camcanCSV = dataDir + 'proc_data/oneCSVToRuleThemAll.csv'
	subjectData = pd.read_csv(camcanCSV)
	subjectData = subjectData[subjectData['numEpochs'] > 55]
	subjectIDs = subjectData['SubjectID'].tolist() 
	numEpochs =  subjectData['numEpochs'].tolist() 
	epoch_dict = dict(zip(subjectIDs, numEpochs))
	subjectAges = subjectData['Age_x'].tolist()
	Ages_dict = dict(zip(subjectIDs, subjectAges))
	print('subject list generated')

	thresholds = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.75, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]


	# integer number of cycles in wavelet (> 5 advisable)
	width = 10

	# vector of frequencies in TFR (measured in integer number of Hz)
	fmin = 15			
	fmax = 30		
	fstep = 1 			
	fVec = np.arange(fmin, fmax+fstep, fstep)

	columns = ['subject_trial', 'threshold', 'coef']
	df_plot = pd.DataFrame(columns=columns)


	for subject in subjectIDs:


		# JUST TO GET Fs
		#################################################################################
		# set epoch directory        
		inDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef'
		epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
		epochFif = os.path.join(inDir, subject, epochFifFilename)

		# read the epochs
		epochs = mne.read_epochs(epochFif)
		Fs = epochs.info['sfreq']
		#################################################################################


		inDir = dataDir 
		file = '_comData.npy'
		path = inDir + subject + file

		if os.path.exists(path):

			tc = np.load(path) # this is array [63, 1, 3401]

			# focus on prestim
			start_time = (-1.25 - -1.5)*1000
			tc = tc[:, :, int(start_time):int(start_time)+1001]

			#get rid of extra axis
			tc = np.squeeze(tc, axis=1) # now TFR has shape [63 x 1001]


			#Make a TFR for each epoch [trials x frequency x time] = [63 x 16 x 1001]
			TFR, tVec, fVec = spectralevents_ts2tfr(tc.T, fVec, Fs, width)

			# make a list of 2d arrays (one for each trial). i.e. this is a list of spectrograms for this subject
			trial_tfrs = []
			for trial in range(TFR.shape[0]):
				trial_tfrs.append(TFR[trial])

			trial = 1
			for spectrogram in trial_tfrs:

				
				median = np.median(spectrogram)
				spectrogram_avg_beta_power = np.mean(spectrogram, axis=0)
		
				for threshold in thresholds:	

					spectrogram_thresholded = np.where(spectrogram > median*threshold, 1, 0)
					percent_of_pixels = percent_pixels(spectrogram_thresholded, spectrogram_thresholded.shape[0])

					# calculate cprrelation coefficient betweehn spectrogram_avg_beta_power and percent_of_pixels
					x = np.corrcoef(spectrogram_avg_beta_power, percent_of_pixels)[1,0]
					subject_trial = str(subject + '-'+ str(trial) )
					values = [subject_trial, threshold, x]
					dictionary = dict(zip(columns, values))
					df_plot = df_plot.append(dictionary, ignore_index=True)

				trial = trial+1
		
		# set output director for png files
		df_plot.to_csv(dataDir + 'source_shin_threshold.csv')	



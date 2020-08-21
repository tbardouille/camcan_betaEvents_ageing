#!/usr/bin/env python

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
import support_functions.spectralevents_functions as tse

import logging
import multiprocessing as mp
import warnings

logger = logging.getLogger(__name__)
mne.set_log_level("WARNING")

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())



def get_spectral_events(subjectID):
    """Top-level run script for finding spectral events in MEG data."""

    # Event-finding method (1 allows for maximal overlap while 2 limits overlap in each respective suprathreshold region)
    #   Only method 1 is implemented so far
    findMethod = 1
    # Factors of Median threshold (see Shin et al. eLife 2017 for details concerning this value)
    thrFOM = 6

    tmin = -1.7     # seconds
    tmax = 1.7     # seconds
    fmin = 1        # Hertz (integer)
    fmax = 105       # Hertz (integer)
    fstep = 1       # Hertz (integer)
    width = 10      # integer number of samples
    LCMV_regularization = 0.5
    channelName = 'betaERD_ROI'

    footprintFreq = 4
    footprintTime = 80
    threshold = 0.00
    neighbourhood_size = (footprintFreq,footprintTime)

    # Setup paths and names for file
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/proc_data/TaskSensorAnalysis_transdef'
    subjectsDir =  homeDir +'/camcan/subjects/'
    groupSourceDir = homeDir + '/camcan/source_data/TaskSensorAnalysis_transdef/fsaverage/'
    outDir = homeDir + '/camcan/'#, subjectID)
    epochFifFilename = 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'

    csvPrefix = "".join([subjectID, '_', channelName, 
        '_', str(tmin), '-', str(tmax), 's', 
        '_', str(fmin), '-', str(fmax), 'Hz', '_', str(fstep),
        'Hzstep_spectral_events'])

    npyPrefix = "".join([channelName, 
        '_', str(tmin), '-', str(tmax), 's'])

    csvFile = os.path.join(outDir, "".join([csvPrefix, ".csv"]))
    logFile = os.path.join(outDir, "".join([csvPrefix, ".log"]))
    npyFile = os.path.join(outDir, "".join([npyPrefix, ".npy"]))

    plotOK = False

    ################################
    # Processing starts here
    logger.info(subjectID)
    # Make the filename with path
    epochFif = os.path.join(dataDir, subjectID, epochFifFilename)
    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'
    funcLabelFile = groupSourceDir + 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo_ERD_DICS_funcLabel-lh.label'

    if not os.path.exists(csvFile):
    
        if os.path.exists(epochFif):

            print(subjectID)
            # Make output file folder
            #if not os.path.exists(outDir):
                #os.makedirs(outDir)

            '''
            # Setup log file for standarda output and error
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(message)s',
                filename=logFile,
                filemode='w'
            )
            '''

            # Read the epochs
            epochs = mne.read_epochs(epochFif)
            Fs = epochs.info['sfreq']

            # Read source space
            src = mne.read_source_spaces(srcFif)
            # Make forward solution
            forward = mne.make_forward_solution(epochs.info,
                                            trans=transFif, src=src, bem=bemFif,
                                            meg=True, eeg=False)

            # Read functional ROI label, morph to subject's MRI and take centre or mass for source estimation
            label = mne.read_label(funcLabelFile)
            label.morph(subject_from='fsaverage', subject_to='sub-' + subjectID, subjects_dir=subjectsDir)

            # Compute LCMV source estimate over epochs at ROI
            noise_cov = mne.compute_covariance(epochs, tmin=-1.7, tmax=-0.2, method='shrunk')
            data_cov = mne.compute_covariance(epochs, tmin=0.0, tmax=1.5, method='shrunk')
            filters = mne.beamformer.make_lcmv(epochs.info, forward, data_cov, reg=LCMV_regularization,
                                           noise_cov=noise_cov, pick_ori='max-power', weight_norm='unit-noise-gain',
                                           label=label)
            stc = mne.beamformer.apply_lcmv_epochs(epochs, filters, max_ori_out='signed')

            # Find center of mass of the LCMV
            stcLabel = mne.Label(stc[0].vertices[0], hemi='lh', subject='sub-' + subjectID)
            a = stcLabel.center_of_mass(subject='sub-' + subjectID, subjects_dir=subjectsDir, restrict_vertices=True)
            comIndex = np.where(stc[0].vertices[0]==a)[0][0]

            # Extract data for the center of mass only
            stcData = []
            for thisStc in stc:
                thisStc.crop(tmin,tmax)
                stcData.append(thisStc.data)
            stcData = np.asarray(stcData)
            epochData = np.squeeze(stcData[:, comIndex, :])

            # Vector of frequencies in TFR [Hz]
            fVec = np.arange(fmin, fmax+fstep, fstep)

            # Make a TFR for each epoch [trials x frequency x time]
            #logger.info('Calculating TFRs')
            TFR, tVec, fVec = tse.spectralevents_ts2tfr(epochData.T, fVec, Fs, width)

            # Vector of times in TFR [s]
            tVec = tVec + tmin

            # Set all class labels to the same value (button press)
            numTrials = TFR.shape[0]
            classLabels = [1 for x in range(numTrials)]

            # Find spectral events based on TFR
            #logger.info('Finding Spectral Events')
            spectralEvents = tse.spectralevents_find (findMethod, thrFOM, tVec,
                fVec, TFR, classLabels, neighbourhood_size, threshold, Fs)
            #logger.info('Found ' + str(len(spectralEvents)) + ' spectral events.')

            # Write data to a CSV file
            #logger.info('Writing event characteristics to file')
            df = pd.DataFrame(spectralEvents)
            df['Subject ID'] = subjectID
            df.to_csv(csvFile)

            #np.save(npyFile, epochData)
            return epochData, TFR, df


if __name__ == "__main__":

    # Find subjects to be analysed
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    camcanCSV = dataDir + 'proc_data/oneCSVToRuleThemAll.csv'
    subjectData = pd.read_csv(camcanCSV)
    
    # Take only subjects with more than 55 epochs
    subjectData = subjectData[subjectData['numEpochs'] > 55]
    subjectIDs = subjectData['SubjectID'].tolist()

    ############ HACK FOR TESTING ##############################################
    subjectIDs = subjectIDs[0:3]
    ############################################################################


    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/3))
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(get_spectral_events, subjectIDs)
    #epochData, TFR, df = get_spectral_events(subjectIDs[0])




    # combine all individual spectral events lists into one main list


    df_all = pd.DataFrame([])

    for subjectID in subjectIDs:

        # for keeping track of run time
        print(subjectID)

        # must read in a separate csv events list for each participant
        csvFile = dataDir + subjectID + '_betaERD_ROI_-1.7-1.7s_1-105Hz_1Hzstep_spectral_events.csv' 


        if os.path.exists(csvFile):

            df = pd.read_csv(csvFile)
            df_all = df_all.append(df)



    outputCSV = 'grand_events_list_source.csv' 
    csvFile = dataDir + outputCSV

    df_all.to_csv(csvFile)

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


def get_spectral_events(subjectID, Ages_dict):

    print(subjectID)
    """Top-level run script for finding spectral events in MEG data."""

    # Event-finding method (1 allows for maximal overlap while 2 limits overlap in each respective suprathreshold region)
    #   Only method 1 is implemented so far
    findMethod = 1  # Method 4 is a potential fix of method 1 (which gives negative durations and freq spans)
    # Factors of Median threshold (see Shin et al. eLife 2017 for details concerning this value)
    thrFOM = 6 


    footprintFreq = 4
    footprintTime = 80
    threshold = 0.00
    neighbourhood_size = (footprintFreq,footprintTime)


    sides = ['left', 'right']


    # Setup paths and names for file
    # (DONT NEED) dataDir = '/home/timb/camcan/proc_data/TaskSensorAnalysis_transdef'

    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'

 
    spectralEventsCSV = "".join([subjectID, '_spectral_events_list.csv']) 
    csvFile = os.path.join(dataDir, spectralEventsCSV)
    
    trials = list(range(64))

    df_subject = pd.DataFrame([])

    for side in sides:

        df_side = pd.DataFrame([])

        for trial in trials:
            

            TFRfname = dataDir + subjectID + '_' + side + '_trial=' + str(trial) + '_frange=5-105Hz_fstep=1Hz-tfr.h5'

            plotOK = False
            

            if os.path.exists(TFRfname):


                #############################################################################################################
                # import AverageTFR object
                TFR_object = mne.time_frequency.read_tfrs(TFRfname)[0]

                #get parameters from the average TFR object
                tVec = TFR_object.times
                fVec = TFR_object.freqs
                Fs = TFR_object.info.get('sfreq') 
                TFR = TFR_object.data # array shape (1,29,3401) = (trials, frequency, time)
                ###################################################################################################################

                # Set all class labels to the same value (button press)
                numTrials = TFR.shape[0] # this should turn out to be = 1
                classLabels = [1 for x in range(numTrials)]
                
                # Find spectral events based on TFR
                #logger.info('Finding Spectral Events')
                spectralEvents = tse.spectralevents_find (findMethod, thrFOM, tVec,
                    fVec, TFR, classLabels, neighbourhood_size, threshold, Fs)
                #logger.info('Found ' + str(len(spectralEvents)) + ' spectral events.')

                # Write data to a CSV file
                #logger.info('Writing event characteristics to file')
                df_trial = pd.DataFrame(spectralEvents)
                df_trial['Subject ID'] = subjectID
                df_trial['side'] = side
                df_trial['Trial'] = trial
                df_trial['Subject Age'] = Ages_dict[subjectID]
                df_side = df_side.append(df_trial)
                #df_trial.to_csv(csvFile)

        df_subject = df_subject.append(df_side)
    df_subject.to_csv(csvFile)

        
    return findMethod, thrFOM, tVec, fVec, TFR, classLabels, neighbourhood_size, threshold, Fs 

if __name__ == "__main__":

    # Find subjects to be analysed
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    camcanCSV = dataDir + 'proc_data/oneCSVToRuleThemAll.csv'
    subjectData = pd.read_csv(camcanCSV)

    # Take only subjects with more than 55 epochs
    subjectData = subjectData[subjectData['numEpochs'] > 55]
    subjectIDs = subjectData['SubjectID'].tolist()
    subjectAges = subjectData['Age_x'].tolist()
    Ages_dict = dict(zip(subjectIDs, subjectAges))


    ############ HACK FOR TESTING ##############################################
    subjectIDs = subjectIDs[0:3]
    ############################################################################

    for subjectID in subjectIDs:
        get_spectral_events(subjectID, Ages_dict)





    # combine all individual spectral events lists into one main list


    df_all = pd.DataFrame([])

    for subjectID in subjectIDs:

        # for keeping track of run time
        print(subjectID)

        # must read in a separate csv events list for each participant
        csvFile = dataDir + subjectID + '_spectral_events_list.csv' 

        if os.path.exists(csvFile):

            df = pd.read_csv(csvFile)
            df_all = df_all.append(df)



    outputCSV = 'grand_events_list_sensor_ROI.csv' 
    csvFile = dataDir + outputCSV

    df_all.to_csv(csvFile)
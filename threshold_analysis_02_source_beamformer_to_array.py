#!/usr/bin/env python

# Import libraries
import os, sys
import numpy as np
import pandas as pd
import mne
import logging
import multiprocessing as mp

# Script to read TFR and PSDS data, grand-average, and write interesting findings to a panda frame

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


def ERD_ROI_TFR(subjectID):

    

    # Settings
    fmin = 10
    fmax = 35
    fstep = 1
    #fmin = 5
    #fmax = 90
    #fstep = 5
    LCMV_regularization = 0.5

    # Define paths
    homeDir = '/home/timb'
    #homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    megDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef/' + subjectID + '/'
    #outDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/' + subjectID + '/'
    outDir = '/media/NAS/bbrady/ROI_Analysis/camcan_ERD_ROI_TFRs/'
    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'
    subjectsDir = dataDir + 'subjects/'
    groupSourceDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/fsaverage/'

    # Make source path if it does not exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Files that exits
    epochFif = megDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'
    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'   
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'
    funcLabelFile = groupSourceDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_ERD_DICS_funcLabel-lh.label'

    # Files to make
    tfrFile1 = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_ERD_ROI_TFR_' + str(fmin) + '-' + str(fmax) + 'Hz'
    tfrFile2 = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_ERD_CofM_TFR_' + str(fmin) + '-' + str(fmax) + 'Hz'
    tfrFile3 = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_ERD_CofM_noEvoked_TFR_' + str(fmin) + '-' + str(fmax) + 'Hz'
    comDataFile = outDir + subjectID  + '_comData'

    if not os.path.exists(comDataFile):
        print(subjectID)

        # Setup log file for standarda output and error
        logFile = outDir + dsPrefix + '_ERD_DICS_ROI_TFR_processing_notes.txt'
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename=logFile,
            filemode='w'
        )
        stdout_logger = logging.getLogger('STDOUT')
        sl = StreamToLogger(stdout_logger, logging.INFO)
        sys.stdout = sl
        stderr_logger = logging.getLogger('STDERR')
        sl = StreamToLogger(stderr_logger, logging.ERROR)
        sys.stderr = sl

        # Read epochs
        epochs = mne.read_epochs(epochFif)
        # Read source space
        src = mne.read_source_spaces(srcFif)
        # Make forward solution
        forward = mne.make_forward_solution(epochs.info,
                                            trans=transFif, src=src, bem=bemFif,
                                            meg=True, eeg=False)

        # Read functional ROI label, morph to subject's MRI and take centre or mass for source estimation
        label = mne.read_label(funcLabelFile)
        label.morph(subject_from='fsaverage', subject_to='sub-' + subjectID, subjects_dir=subjectsDir)

        # Compute LCMV time-frequency response at ROI
        noise_cov = mne.compute_covariance(epochs, tmin=-1.7, tmax=-0.2, method='shrunk')
        data_cov = mne.compute_covariance(epochs, tmin=0.0, tmax=1.5, method='shrunk')
        filters = mne.beamformer.make_lcmv(epochs.info, forward, data_cov, reg=LCMV_regularization,
                                           noise_cov=noise_cov, pick_ori='max-power', weight_norm='unit-noise-gain',
                                           label=label)
        stc = mne.beamformer.apply_lcmv_epochs(epochs, filters, max_ori_out='signed')

        # Make a label based on the LCMV, then pull the center of mass
        stcLabel = mne.Label(stc[0].vertices[0], hemi='lh', subject='sub-' + subjectID)
        a = stcLabel.center_of_mass(subject='sub-' + subjectID, subjects_dir=subjectsDir, restrict_vertices=True)
        comLabel = mne.Label([a], hemi='lh', subject='sub-' + subjectID)

        # Calculate LCMV beamformer at centre of mass
        filters2 = mne.beamformer.make_lcmv(epochs.info, forward, data_cov, reg=LCMV_regularization,
                                            noise_cov=noise_cov, pick_ori='max-power', weight_norm='unit-noise-gain',
                                            label=comLabel)
        stc2 = mne.beamformer.apply_lcmv_epochs(epochs, filters2, max_ori_out='signed')
        #print(stc2)

        # TFR Analysis Starts Here
        epochTimes = epochs.copy()
        times = epochTimes.crop(tmin=-1.5, tmax=1.5).times
        sfreq = epochs.info['sfreq']
        freqs = np.arange(fmin, fmax+fstep, fstep)
        n_cycles = freqs / 3.  # different number of cycle per frequency

  
        # Now put centre of mass beamformer data into an array and save
        comData = []
        for thisSTC in stc2:
            comData.append(thisSTC.data)
        comData = np.asarray(comData)
        np.save(comDataFile, comData)






if __name__ == '__main__':

    # Find subjects to be analysed
    homeDir = '/home/timb'
    dataDir = homeDir + '/camcan/'
    evokedStatsCSV = dataDir + 'source_data/ERD_stats.csv'
    subjectData = pd.read_csv(evokedStatsCSV)

    # Take only subjects with more than 60 epochs
    subjectData2 = subjectData.copy()
    subjectData2 = subjectData2.loc[subjectData2['ERDstcMorphExists']]
    numSubjects = len(subjectData2)
    subjectIDs = subjectData2['SubjectID'].tolist()
    

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/2))
    print(count)
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(ERD_ROI_TFR, subjectIDs)
	


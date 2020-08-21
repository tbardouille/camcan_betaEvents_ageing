# camcan_spectralEvents_ageing
This repository contains files used in the NeuroImage paper entitled: Age-Related Trends in Neuromagnetic Transient Beta Burst Characteristics During a Sensorimotor Task and Rest in the Cam-CAN Open-Access Dataset

The order to run scripts is indicated by the number in the file name. 

spectralevents_functions.py does not have to be run but contains support functions for the other scripts.

Rest data analysis is omitted in these script as the basic code structure is the same for the task data.

Note that 07_ageing_trends.py generates 2 scatterplots for each phase/characteristic combo (one has a linear fit and one has a quadratic fit). The statistical analysis (F-test - contained within the same script) determines which fit is best.

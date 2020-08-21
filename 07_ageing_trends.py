# Import libraries
import os
import sys
import time
import mne
import math
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
from scipy import stats



def separate_by_phase(fmin, fmax, tmin, tmax, df_task):
	df = df_task.copy()
	df_phase = df[( (df['Peak Time'] >= tmin) & (df['Peak Time'] <= tmax) & (df['Peak Frequency'] >= fmin) & (df['Peak Frequency'] <= fmax) )]
	return df_phase

def calc_stat(subject, df_stat):
	df = df_stat.copy()
	df = df.loc[df['Subject ID'] == subject]
	num_of_events=len(df)
	df_means = df.mean()
	df_std = df.std()
	return df_means, df_std, num_of_events


def threshold(df_og, thr):
	df = df_og.copy()		
	df['outlier_thr'] = df['Normalized Peak Power'] > thr
	df = df[df['outlier_thr']]
	return df


if __name__ == "__main__":

	#Frequency Bands (Hz)
	theta_min = 4
	theta_max = 8
	mu_min = 8
	mu_max = 15
	beta_min = 15
	beta_max = 30
	gamma_min = 60
	gamma_max = 95

	#Oscillatory Correlates Start and End times (s)
	Baseline_tmin = -1.5
	Baseline_tmax = -1.0

	PreM_tmin = -1.25
	PreM_tmax = -0.25

	M_tmin = -0.25	
	M_tmax = 0.25	

	gammaM_tmin = -0.2
	gammaM_tmax = 0.2

	PostM_tmin = 0.25 	
	PostM_tmax = 1.25 	

	Rest_tmin = -100
	Rest_tmax = 100

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


	############ HACK FOR TESTING ##############################################
	subjectIDs = subjectIDs[0:3]
	############################################################################


	# Generate Sensor_ROI Task Data and take only ourlier events
	homeDir = os.path.expanduser("~")
	sourceDir = homeDir + '/camcan/'
	csvFile = 'grand_events_list_sensor_ROI.csv'
	csvData = os.path.join(sourceDir, csvFile)
	df_senROI_t = pd.read_csv(csvData)
	df_senROI_L = df_senROI_t.copy()
	df_senROI_t_L = df_senROI_L[df_senROI_L['side'] == 'left']
	df_senROI_t_L = threshold(df_senROI_t_L, 1.8)
	print('sensor ROI task data generated')
	

	# Generate Source Task Data [for ERD] and take only ourlier events
	homeDir = os.path.expanduser("~")
	sourceDir = homeDir + '/camcan/'
	csvFile = 'grand_events_list_source.csv'
	csvData = os.path.join(sourceDir, csvFile)
	df_sou_t_L_ERD = pd.read_csv(csvData)
	df_sou_t_L_ERD = threshold(df_sou_t_L_ERD, 2.1)
	print('source task data generated (ERD)')

	
	# Generate Single Sensor Task Data and take only ourlier events
	homeDir = os.path.expanduser("~")
	sourceDir = homeDir + '/camcan/'
	csvFile = 'grand_events_list_source.csv'
	csvData = os.path.join(sourceDir, csvFile)
	df_Ssen_t_L = pd.read_csv(csvData)
	df_Ssen_t_L = threshold(df_Ssen_t_L, 2.3)
	print('single sensor task data generated')


	# set output director for png files
	outDir = homeDir + '/camcan/' 


	# make a separate dataframe for side-phase combo

	df_senROI_beta_PreM = separate_by_phase(beta_min, beta_max, PreM_tmin, PreM_tmax, df_senROI_t_L)
	df_senROI_beta_M= separate_by_phase(beta_min, beta_max, M_tmin, M_tmax, df_senROI_t_L)       
	df_senROI_beta_PostM = separate_by_phase(beta_min, beta_max, PostM_tmin, PostM_tmax, df_senROI_t_L)
	
	df_source_beta_PreM = separate_by_phase(beta_min, beta_max, PreM_tmin, PreM_tmax, df_sou_t_L_ERD)
	df_source_beta_M= separate_by_phase(beta_min, beta_max, M_tmin, M_tmax, df_sou_t_L_ERD)       
	df_source_beta_PostM = separate_by_phase(beta_min, beta_max, PostM_tmin, PostM_tmax, df_sou_t_L_ERD) # note: in the paper, PMBR localization was used for this

	df_SINsen_beta_PreM = separate_by_phase(beta_min, beta_max, PreM_tmin, PreM_tmax, df_Ssen_t_L)
	df_SINsen_beta_M= separate_by_phase(beta_min, beta_max, M_tmin, M_tmax, df_Ssen_t_L)       
	df_SINsen_beta_PostM = separate_by_phase(beta_min, beta_max, PostM_tmin, PostM_tmax, df_Ssen_t_L)
	

	dfs_senROI = [df_senROI_beta_PreM, df_senROI_beta_M, df_senROI_beta_PostM]
	dfs_source = [df_source_beta_PreM, df_source_beta_M, df_source_beta_PostM]  
	dfs_SINsen = [df_SINsen_beta_PreM, df_SINsen_beta_M, df_SINsen_beta_PostM]


	dfs_all = [dfs_SINsen, dfs_senROI, dfs_source]              
	
	m = 0

	for df_method in dfs_all:

		# which phase are we talking about
		if m == 0:
			method = 'Single_Sensor'
		if m == 1:
			method = 'Sensor_ROI'
		if m == 2:
			method = 'Source'

		# make an empty dataframe
		columns = ['Subject ID', 'Subject Age', 'Phase', 'Number of Events', 'Burst Rate', 'Event Duration Mean', 'Event Duration STD', 'NormalizedPeakPowerMean', 'Normalized Peak Power STD', 'Peak Frequency Mean', 'Peak Frequency STD', 'Frequency Span Mean', 'Frequency Span STD']
		df_avgs = pd.DataFrame(columns=columns)


		# for each of the side-phase dataframes, calc average event duration
		i=0
		for df in df_method:

			if i == 0:
				phase = 'Pre-Movement'
				t_range = PreM_tmax - PreM_tmin
			if i == 1:
				phase = 'Movement'
				t_range = M_tmax - M_tmin 
			if i == 2:
				phase = 'Post-Movement'
				t_range = PostM_tmax - PostM_tmin



			# calc avg (and std) burst charactertistics for each subject
			for subject in subjectIDs:
				print(method + '_' + phase + '_' + subject)
				age = float(df_senROI_t_L[df_senROI_t_L['Subject ID']==subject]['Subject Age'].values[0])
				means, std, num_of_events = calc_stat(subject, df)

				BR = num_of_events / (epoch_dict[subject] * t_range)

				values = [subject, age, phase, num_of_events, BR, means['Event Duration'], std['Event Duration'], means['Normalized Peak Power'], std['Normalized Peak Power'], means['Peak Frequency'], std['Peak Frequency'], means['Frequency Span'], std['Frequency Span']]
				dictionary = dict(zip(columns, values))
				df_avgs = df_avgs.append(dictionary, ignore_index=True)
				df_avgs['Burst Rate'] = df_avgs['Burst Rate'].replace({0:np.nan})
				df_avgs.NormalizedPeakPowerMean=df_avgs.NormalizedPeakPowerMean.where(df_avgs.NormalizedPeakPowerMean.between(0,10))
				#df_avgs[df_avgs['Normalized Peak Power Mean'] > 100] = 'NaN' ##.loc[df_avgs['Normalized Peak Power Mean'] == 'NaN', 'Normalized Peak Power Mean'] > 100 # = df_avgs['Normalized Peak Power Mean'].replace({0:np.nan})

			i=i+1





		####################################################################################################################
		##############################################    GENERATE PLOTS    ################################################
		####################################################################################################################

		sns.set(style='white', palette='deep' ,font_scale=2.5)


		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Burst Rate", fit_reg=True, order=2, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'orangered'})#'limegreen'}) 
		axes = plt.gca()
		axes.set_ylim([1,4.5])
		g.set(xticks=[])
		g.set(yticks=[1,3,5])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)
		plt.savefig(outDir + method + '_level_Burst_Rate_with_Age_QUAD.png', bbox_inches='tight')
		print(method + '_level_Burst_Rate_with_Age.png saved to '+ outDir )
		plt.close()
		

		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Burst Rate", fit_reg=True, order=1, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'limegreen'})#'limegreen'}) 
		axes = plt.gca()
		axes.set_ylim([1,4.5])
		g.set(xticks=[])
		g.set(yticks=[1,3,5])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)
		plt.savefig(outDir + method + '_level_Burst_Rate_with_Age_LIN.png', bbox_inches='tight')
		print(method + '_level_Burst_Rate_with_Age.png saved to '+ outDir )
		plt.close()


		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Event Duration Mean", fit_reg=True, order=2, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'orangered'}) 
		axes = plt.gca()
		axes.set_ylim([0.17,0.3])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)		
		plt.savefig(outDir + method + '_level_Event_Duration_with_Age_QUAD.png', bbox_inches='tight')
		print(method + '_level_Event_Duration_with_Age.png saved to '+ outDir )
		plt.close()

		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Event Duration Mean", fit_reg=True, order=1, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'limegreen'}) 
		axes = plt.gca()
		axes.set_ylim([0.17,0.3])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)		
		plt.savefig(outDir + method + '_level_Event_Duration_with_Age_LIN.png', bbox_inches='tight')
		print(method + '_level_Event_Duration_with_Age.png saved to '+ outDir )
		plt.close()

		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "NormalizedPeakPowerMean", fit_reg=True, order=2, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'orangered'}) 
		axes = plt.gca()
		axes.set_ylim([3.5,10])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)		
		plt.savefig(outDir + method + '_level_Normalized_Peak_Power_with_Age_QUAD.png', bbox_inches='tight')
		print(method + '_level_Normalized_Peak_Power_with_Age.png saved to '+ outDir )
		plt.close()

		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "NormalizedPeakPowerMean", fit_reg=True, order=1, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'limegreen'}) 
		axes = plt.gca()
		axes.set_ylim([3.5,10])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)		
		plt.savefig(outDir + method + '_level_Normalized_Peak_Power_with_Age_LIN.png', bbox_inches='tight')
		print(method + '_level_Normalized_Peak_Power_with_Age.png saved to '+ outDir )
		plt.close()


		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Peak Frequency Mean", fit_reg=True, order=2, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'orangered'}) 
		axes = plt.gca()
		axes.set_ylim([21,25])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)	
		plt.savefig(outDir + method + '_level_Peak_Frequency_with_Age_QUAD.png', bbox_inches='tight')
		print(method + '_level_Peak_Frequency_with_Age.png saved to ' + outDir )
		plt.close()


		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Peak Frequency Mean", fit_reg=True, order=1, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'limegreen'}) 
		axes = plt.gca()
		axes.set_ylim([21,25])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)	
		plt.savefig(outDir + method + '_level_Peak_Frequency_with_Age_LIN.png', bbox_inches='tight')
		print(method + '_level_Peak_Frequency_with_Age.png saved to ' + outDir )
		plt.close()


		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Frequency Span Mean", fit_reg=True, order=2, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'orangered'}) 
		axes = plt.gca()
		axes.set_ylim([6.5,9.5])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)	
		plt.savefig(outDir + method + '_level_Frequency_Span_with_Age_QUAD.png', bbox_inches='tight')
		print(method +  '_level_Frequency_Span_with_Age.png saved to ' + outDir )
		plt.close()


		g = sns.FacetGrid(df_avgs, col="Phase", aspect=1.5, gridspec_kws={"wspace":0.1})
		g.map(sns.regplot, "Subject Age", "Frequency Span Mean", fit_reg=True, order=1, scatter_kws={'s':10}, line_kws={'linewidth':3.5, 'color':'limegreen'}) 
		axes = plt.gca()
		axes.set_ylim([6.5,9.5])
		g.set(xticks=[])
		titles = g.axes.flatten()
		titles[0].set_title("")
		titles[1].set_title("")
		titles[2].set_title("")
		titles[0].set_ylabel("")
		titles[0].set_xlabel("")
		titles[1].set_xlabel("")
		titles[2].set_xlabel("")
		for ax in titles:
			ax.tick_params(axis='y', colors='dimgrey')
			for _, spine in ax.spines.items():
				spine.set_visible(True)
				spine.set_color('dimgrey')
				spine.set_linewidth(2)
		plt.xlim(0,106)	
		plt.savefig(outDir + method + '_level_Frequency_Span_with_Age_LIN.png', bbox_inches='tight')
		print(method +  '_level_Frequency_Span_with_Age.png saved to ' + outDir )
		plt.close()



		####################################################################################################################
		##########################################    Statistical Analyses    ##############################################
		####################################################################################################################



		#make lists to iterate through
		df_avgs_PreM = df_avgs.loc[df_avgs['Phase'] == 'Pre-Movement']
		df_avgs_M = df_avgs.loc[df_avgs['Phase'] == 'Movement']
		df_avgs_PostM = df_avgs.loc[df_avgs['Phase'] == 'Post-Movement']
		df_list = [df_avgs_PreM, df_avgs_M, df_avgs_PostM]

		characteristics = ['Burst Rate', 'Event Duration Mean', 'NormalizedPeakPowerMean', 'Peak Frequency Mean','Frequency Span Mean']

		# make an empty dataframe
		columns = ['characteristic', 'phase', 'slope', 'intercept', 'r_value', 'p_value', 'std_err']
		df_stats = pd.DataFrame(columns=columns)

		#for polyfit
		columns_2 = ['characteristic', 'phase', 'quad', 'linear', 'intercept', 'residuals', 'rank', 'singular_values', 'rcond']
		df_stats_2 = pd.DataFrame(columns=columns_2)

		#for polyfit again
		columns_3 = ['characteristic', 'phase', 'quad', 'linear', 'intercept', 't_stat_quad', 't_stat_linear', 't_stat_intercept']
		df_stats_3 = pd.DataFrame(columns=columns_3)

		#for Ftest
		columns_F = ['characteristic', 'phase', 'F', 'chi_squared_min_linear', 'chi_squared_min_quad']
		df_stats_F = pd.DataFrame(columns=columns_F)

		i=0
		for df in df_list:

			if i == 0:
				phase = 'Pre-Movement'
				t_range = PreM_tmax - PreM_tmin
			if i == 1:
				phase = 'Movement'
				t_range = M_tmax - M_tmin 
			if i == 2:
				phase = 'Post-Movement'
				t_range = PostM_tmax - PostM_tmin


			for characteristic in characteristics:
				df = df.dropna(axis=0)
				x = df['Subject Age']
				y = df[characteristic]
				keys = columns
				keys_2 = columns_2
				keys_3 = columns_3

				#linear fit
				slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
				values = [characteristic, phase, slope, intercept, r_value, p_value, std_err]
				dictionary = dict(zip(keys, values))
				df_stats = df_stats.append(dictionary, ignore_index=True)

				# poly fit
				degree = 2
				coefs, residuals, rank, singular_values, rcond = np.polyfit(x, y, degree, full=True)
				values_2 = [characteristic, phase, coefs[0], coefs[1], coefs[2], residuals, rank, singular_values, rcond]
				dictionary_2 = dict(zip(keys_2, values_2))
				df_stats_2 = df_stats_2.append(dictionary_2, ignore_index=True)

				# poly fit again for quadratic fit variances
				coefs, cov_matrix = np.polyfit(x, y, degree, full=False, cov='unscaled')
				t_stat_quad = coefs[0]/math.sqrt(abs(cov_matrix[0,0]))
				t_stat_linear = coefs[1]/math.sqrt(abs(cov_matrix[1,1]))
				t_stat_intercept = coefs[2]/math.sqrt(abs(cov_matrix[2,2]))
				values_3 = [characteristic, phase, coefs[0], coefs[1], coefs[2], t_stat_quad, t_stat_linear, t_stat_intercept]
				dictionary_3 = dict(zip(keys_3, values_3))
				df_stats_3 = df_stats_3.append(dictionary_3, ignore_index=True)


				if characteristic == 'Burst Rate':
					characteristic_std = 'Burst Rate STD'

				if characteristic == 'Event Duration Mean':
					characteristic_std = 'Event Duration STD'

				if characteristic == 'NormalizedPeakPowerMean':
					characteristic_std = 'Normalized Peak Power STD'

				if characteristic == 'Peak Frequency Mean':
					characteristic_std = 'Peak Frequency STD'

				if characteristic == 'Frequency Span Mean':
					characteristic_std = 'Frequency Span STD'

				chi_squared_min_linear = 0
				chi_squared_min_quad = 0

				# F-test (to determine if quad or linear is best)
				for age in range(18,89):
					
					############# find measured characteristic and variance 
					df_age = df[df['Subject Age'] == age]
					k = len(df_age)


					if k>0:
						df_age['Burst Rate STD'] = np.sqrt(df_age['Burst Rate']) #an educated guess of the std of burst rate
						df_age = df_age[[characteristic, characteristic_std]]

						# calculate variances from standard deviations
						df_age[characteristic + 'VAR'] = df_age[characteristic_std]*df_age[characteristic_std]

						# calculate the mean of each characteristics and the variance of the mean
						char_avg = df_age.mean(axis=0) #means for each characteristic
						# to calculate a variance of a mean of means (each with a variance): var_of_mean = sum(variances)/N^2 = (avg variance)/N,  (N = total number of subjects with age=age)
						# so, divide average varainces by N
						if k == 0:
							k=1

						char_avg[characteristic + 'VAR'] = char_avg[characteristic + 'VAR']  / k

						#### predict values of characteristics based on fits
						#linear
						pred_lin = slope*age + intercept
						#quad					
						pred_quad = coefs[0]*age*age + coefs[1]*age + coefs[2]

						### Calc chi squared in loop (linear)
						chi_s = ((char_avg[0] - pred_lin)**2)/(  (char_avg[2])**2 )
						chi_squared_min_linear = chi_squared_min_linear + chi_s

						### Calc chi squared in loop (Quad)
						chi_s_q = ((char_avg[0] - pred_quad)**2)/(  (char_avg[2])**2 )
						chi_squared_min_quad = chi_squared_min_quad + chi_s_q

				p = 1
				N = 88-18
				em = 2
				F = ((chi_squared_min_linear-chi_squared_min_quad)/p)/((chi_squared_min_quad)/(N-em))
				keys_F = columns_F
				values_F = [characteristic, phase, F, chi_squared_min_linear, chi_squared_min_quad]
				dictionary_F = dict(zip(keys_F, values_F))
				df_stats_F = df_stats_F.append(dictionary_F, ignore_index=True)

			i=i+1

		df_stats.to_csv(outDir + method + '_level_age_related_linear_regression_stats.csv')
		df_stats_2.to_csv(outDir + method + '_level_age_related_poly_regression_stats.csv')
		df_stats_F.to_csv(outDir + method + '_level_F_stats.csv')
		df_stats_3.to_csv(outDir + method + '_level_age_related_poly_regression_stats_covariance.csv')
		m=m+1






		


from EINSfit import EINSfit  #import EINSfit class from EINfit.py file

############################
# Minimal examples for 1 or 2 data sets
#####

#####
# Ex1: 1 data set loaded from elascan file called EISF_sample1_q.dat and EISF_sample1_t.dat
my_sample=EINSfit('EISF_sample1')
my_sample.save_config_file()  #creates default config file
#my_sample.read_config_file() #reads the created config file above
my_sample.run_fit()           # runs the sample with the parameters given by config file
my_sample.plot_results()      #show plots of results
my_sample.save_all()          #save results
#####

#####
# Ex2: 2 data sets with config file = config-all.ini
my_samples=[]
results_dic=[]
sample_names=['EISF_sample1','EISF_sample2']
for sample in my_samples:
    sample.append(EINSfit(sample))
    sample[-1].read_config_file('config-all.ini')
    sample[-1].run_fit()
    sample[-1].save_all()
#save ordered results in list of dictionaries
for sample in my_samples:
    results_dic.append(sample.get_nice_results_dic())
#####
############################

############################
# examples to compare results of 2 data sets (from above created dictionary)
#####

import matplotlib.pyplot as plt
color_l=['blue','red']

#####
# plot 1, MSD or STD of different models
plt.figure()
model='GA'  # GA, PK, Yi or Do
para='MSD3' # MSD3 or STD3
for i,dic in enumerate(results_dic):
    plt.errorbar(dic['EISF_T'],dic[para][model]['vals'],dic[para][model]['errors'],label=dic['name'], color=color_l[i])
plt.title(model)
plt.legend()
#####

#####
# plot 2, compare fits at same/similar temperature
from EINSfit import take_closest_value #gives you index of value, which is closest to the desired value
import numpy as np

plt.figure()
t_wanted=360 # desired temperature value
model='PK'   # GA, PK, Yi or Do
q=np.linspace(0,5,100) # x axis
for i,sample in enumerate(my_samples):
    t_used,t_idx=take_closest_value(sample.used_T,t_wanted)
    plt.errorbar(sample.used_q,sample.used_data[t_idx,:],sample.used_data_err[t_idx,:],label=sample.name,color=color_l[i])
    plt.plot(q,sample.give_fit_value(q,t=t_idx,model=model),label='T=%iK'%t_used,color=color_l[i])
plt.title(model)
plt.legend()
#####
############################

############################
# Additional examples
#####

#personal save direcory
my_save_path='all-data'

#####
# A.Ex1: Change input data set:

# exampe for dic_data_to_use with no changes, values have to be floats
dic_data_to_use_example={'T_start': None, 'T_end': None, 'Q_min': None, 'Q_max': None, 'delete_specific_T-values_list': None, 'delete_specific_Q-values_list': None} 
# example for dic_data_to_use with all temperature values, but without Q values smaller than 0.48A-1 and the maximal Q value allowed is 4.5A-1,
#                                  and Q value 3.1252 A-1 is deleted
dic_data_to_use_example2={'Q_min': 0.48, 'Q_max': 4.5, 'delete_specific_Q-values_list': [3.1252,],} 

my_sample=EINSfit('EISF_sample1',name='sample1',data_type='elascan',dic_data_to_use=dic_data_to_use_example,save_dir_path=my_save_path)
#####

#####
# A.Ex2: load saved data set (only raw/used data and if wanted config file, but NOT results!)
my_sample=EINSfit(datafile=my_save_path,name='loaded sample1',data_type='save')
my_sample.read_config_file() #loads saved config file if wanted
#####

#####
# A.Ex3: data set created by user
data_example={'raw_data': np.array([[1,1],[2,2],[3,3]]),'raw_data_err': np.array([[1,1],[2,2],[3,3]]),'raw_T':np.array([100,200,300]),'raw_q':np.array([0.5,1.5])}
my_sample=EINSfit(datafile=data_example,name='created data',data_type='numpy_dic')
#####
############################
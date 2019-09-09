#script now works with python v2 and v3 
#for python 2(tested with v2.7): and needs numpy (tested with 1.14.0),matplotlib (tested with 2.1.2) and lmfit v.0.9.9 packages! with conda (lmfit needs conda-forge), ipython can be added: conda create  --name p27_for_fit_EINS python=2 numpy matplotlib lmfit=0.9.9
#was developed in python2 and is now used with python3 (tested with 3.7.3): and needs numpy (tested with 1.16.3),matplotlib (tested with 3.1.0) and lmfit v.0.9.13 packages! with conda (lmfit needs conda-forge), ipython can be added: conda create  --name p37_for_fit_EINS python=3 numpy matplotlib lmfit=0.9.13
#for python2 support
from __future__ import print_function

import os #for creating directory

#for def readIN_elascan()
import numpy as np
#for exit
import sys
import pickle #too save and load pickles
import copy

import matplotlib as mpl
#matplotlib.use('Agg') #for plotting without display
import matplotlib.pyplot as plt

#for reload
import EINSfit
from importlib import reload
reload(EINSfit)
from EINSfit import EINSfit


#for take closest
from bisect import bisect_left
def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0],pos
    if pos == len(myList):
        return myList[-1],pos-1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after,pos
    else:
       return before,pos-1

dic_data_to_use_example={'T_start':0, 'T_end': 3, 'Q_min':0, 'Q_max': None, 'delete_specific_Q-values_list': None,}
data_example={'raw_data': np.array([[1,1],[2,2],[3,3]]),'raw_data_err': np.array([[1,1],[2,2],[3,3]]),'raw_T':np.array([100,200,300]),'raw_q':np.array([0.5,1.5])}
'''
mydata=EINSfit('../EISF_Thermolysin-D2O')
#mydata.define_fitting_regions()
mydata.run_fit()
mydata.read_config_file(filename='test.ini')
mydata.run_fit()
'''
#test
# mydata=[]
# results_dic=[]
# data_folder_dir='../../'
# samples=[data_folder_dir+'EISF_Thermolysin-D2O',data_folder_dir+'EISF_Thermolysin-GLUCOSE',data_folder_dir+'EISF_Thermolysin-GLY']
# #samples=[data_folder_dir+'EISF_Casein-D2O',data_folder_dir+'EISF_Casein-GLUCOSE',data_folder_dir+'EISF_Casein-GLY']
# #samples=[data_folder_dir+'EISF_Sorbitol',]
# dic_data_to_use_input=None#{'Q_min':1}
# for sample in samples:
#     mydata.append(EINSfit(sample,save_dir_path='data/save_data_all',dic_data_to_use=dic_data_to_use_input))
#     mydata[-1].read_config_file('data/test.ini')
#     mydata[-1].run_fit()
#     mydata[-1].save_all()
# for sample in mydata:
#     results_dic.append(sample.get_nice_results_dic(silent=True))

mydata2=[]
results_dic2=[]
samples_names_Th=['EISF_Thermolysin-D2O','EISF_Thermolysin-GLUCOSE','EISF_Thermolysin-GLY']
samples_names_Ca=['EISF_Casein-D2O','EISF_Casein-GLUCOSE','EISF_Casein-GLY']
#samples_name=samples_names_Th+samples_names_Ca+['EISF_Sorbitol',]
samples_name=['EISF_Sorbitol',]
for sample in samples_name:
    mydata2.append(EINSfit('data/save_data_all',name=sample,data_type='save')) #save_dir_path='first_test_qmin1_load',dic_data_to_use={'Q_min': 1}
    #mydata2[-1].read_config_file() #no name reads saved config = 'datapath/config_file_%s.ini'%self.name
    #mydata2[-1].run_fit()
    #mydata2[-1].set_save_dir('data/save_data_all_01')
    #mydata2[-1].save_all()
for sample in mydata2:
    results_dic2.append(sample.get_nice_results_dic(silent=True))

# overall_dic={}
# for dic in results_dic2:
#     overall_dic[dic['name']]=dic

# mydata3=[]
# samples_name=['EISF_Sorbitol',]
# for sample in samples_name:
#     mydata3.append(EINSfit('data/save_data_Qmin1',name=sample,data_type='save')) #save_dir_path='first_test_qmin1_load',dic_data_to_use={'Q_min': 1}
#     mydata3[-1].read_config_file() #no name reads saved config = 'datapath/config_file_%s.ini'%self.name
#     mydata3[-1].run_fit()
#     #mydata3[-1].save_all()
# for sample in mydata3:
#     results_dic2.append(sample.get_nice_results_dic(silent=True))
#     results_dic2[-1]['name']=results_dic2[-1]['name']+'_Qmin1'



# color_l=['blue','red','black']
# #plotting
# model='GA'
# para='MSD3'
# for i,dic in enumerate(results_dic2):
#     plt.errorbar(dic['EISF_T'],dic[para][model]['vals'],dic[para][model]['errors'],label=dic['name'])
# plt.title(model)
# plt.legend()

# #plotting2
# plt.figure()
# t_wanted=360
# model='PK'
# q=np.linspace(0,5,100)
# for i,fit in enumerate(mydata2):
#     t_used,t_idx=takeClosest(fit.used_T,t_wanted)
#     plt.errorbar(fit.used_q,fit.used_data[t_idx,:],fit.used_data_err[t_idx,:],label=fit.name,color=color_l[i])
#     plt.plot(q,fit.give_fit_value(q,t=t_idx,model=model),label='T=%iK'%t_used,color=color_l[i])
# plt.title(model)
# plt.legend()


# #plotting3 - sorbitol!
# mpl.rcdefaults()
# #python3 code
# with open("plot-parameters.py") as f:
#     code = compile(f.read(), "plot-parameters.py", 'exec')
#     exec(code)#, global_vars, local_vars)

# color_list=['b','r','g',]#'k']
# #color_list=['b','r','g']*2
# color_list=color_list*2
# #color_list=[plt.get_cmap('tab10')(0),plt.get_cmap('tab10')(1),plt.get_cmap('tab10')(2),'k']
# #marker_list=['<','>','^','*']
# marker_list=['o','^','s']#+['*']
# marker_list=marker_list+['*','v','d']
# markerfacecolor=color_list+['None','None','None'] #
# linestyle=['-']*1+[':']*1#['-']*3+[':']*3 #+['--']+
# elinewidth=0.5*plt.rcParams['lines.linewidth']
# plt.rcParams['font.size'] =  12
# plt.rcParams['axes.titlesize'] =  plt.rcParams['font.size']
# plt.rcParams['xtick.labelsize'] =  plt.rcParams['font.size']
# plt.rcParams['ytick.labelsize'] =  plt.rcParams['font.size']
# plt.rcParams['axes.labelsize'] =  plt.rcParams['font.size']
# plt.rcParams['legend.fontsize'] =  plt.rcParams['font.size'] #plt.rcParams['font.size'] #6 for offset plots

# fig_size=(6.69,6.69*1/4.*2+0.25)
# fig1,ft_l=plt.subplots(1,1,figsize=fig_size,sharex='col',sharey='row') #initial sharing

# #plotting
# ft=ft_l
# fig=fig1
# model='GA'
# para='MSD3'
# #for c,dic in enumerate(results_dic2):
# dic=results_dic2[0]
# for c,model in enumerate(['GA','PK']):
#     T=dic['EISF_T']
#     yS=dic[para][model]['vals']
#     yS_Err=dic[para][model]['errors']
#     name=model+' model'
#     ft.errorbar(T,yS,yerr=yS_Err,color=color_list[c],marker=marker_list[c]
#                     ,linestyle = linestyle[c], elinewidth=elinewidth
#                     ,markerfacecolor=markerfacecolor[c]
#                     ,label='%s'%(name))#label=key)
#     handles,labels = ft .get_legend_handles_labels()
#     ft.legend(handles,labels,loc=['best','upper center'][1],numpoints=1,ncol=2)
#     ft.set_ylabel(r'$\left< r^2 \right> [\mathrm{\AA}^2]$')
# ft.set_xlabel('T [K]')
# t_name='Sorbitol'#dic['name']
# ft.set_title('%s'%t_name)
# fig.tight_layout()
# #fig.subplots_adjust(hspace=0.1)
# dir_path='../all/'
# fig.savefig(dir_path+'MSD_%s_%s.pdf'%(t_name,'both-models'),dpi=600)
# fig.savefig(dir_path+'MSD_%s_%s.png'%(t_name,'both-models'),dpi=600)
# plt.show()

# mpl.rcdefaults()
'''
if print_output == True:
    for i,fig1 in enumerate(fits_allT_list):
        fig1.savefig(save_location+'fits_allT-%02d' % (i) +str_fixed+filename_save_str+'.png', dpi=200)
    lin_fig_fit_region.savefig(save_location+'lin_fig_fit_region-'+str_fixed+filename_save_str+'.png', dpi=200)
    figSmith10.savefig(save_location+'testSmith10-'+str_fixed+filename_save_str+'.png', dpi=200)
    if 'figLin_for_doster' in locals(): figLin_for_doster.savefig(save_location+'testDosterLinFit-'+str_fixed+filename_save_str+'.png', dpi=200)
    
    with open(save_location+'options_'+str_fixed+filename_save_str+'.txt',"w") as f:
        f.write('#options:\n')
        for key in options_dic:
            f.write( key+ ' = ' + str(options_dic[key]) + '\n')
    
#save inserted data
save_location_input_data=save_location+'python-data/'
create_dir(save_location_input_data)
save_data(save_location_input_data+'options_dic',options_dic,plain=False) #load with options_dic=np.load('location/options_dic.npy').item() !
save_pickle(options_dic,save_location_input_data+'options_dic.pickle')#pickle
save_data(save_location_input_data+'raw_data_val-'+filename_save_str,p4_DataT_value,plain=True, numpyFormat=True)
save_data(save_location_input_data+'raw_data_err-'+filename_save_str,p4_DataT_error)
save_data(save_location_input_data+'raw_data_Q-'+filename_save_str,p4_Qread)
save_data(save_location_input_data+'raw_data_T-'+filename_save_str,p4_T)
save_data(save_location_input_data+'used_data_val-'+filename_save_str,multi_data)
save_data(save_location_input_data+'used_data_err-'+filename_save_str,multi_dataErr)
save_data(save_location_input_data+'used_data_Q-'+filename_save_str,x)
save_data(save_location_input_data+'used_data_T-'+filename_save_str,str_temp)

print ('-----')
if options_dic['fix_to_offset']:
    print ('Values were fixed to offset of GA')
if options_dic['no_weighting_in_fit']:
    print ('Fit was NOT weighted by errors!')
    if options_dic['no_data_error']:
        print ('... because there was no error given for data!')
else:
    print ('Fit was weighted by errors!')
print ('-----\n\"'+ str(filename)+'\" finished and saved in folder \"%s\" with suffix \"%s\".\n'%(save_location,filename_save_str) + '-'*40 )
'''


''' Example of nicer plot
fig11 = plt.figure(figsize=(5+0.2,5+0.2))
nbplots=0
for i in range(np.maximum(0,len(used_T)-3),len(used_T),2):
    nbplots=nbplots+1
    GA_y_fit = lin_modelLmfitMultiBasic(dic_lmfit['GA'][i].params, i, x_long**2)
    PK_y_fit = pk_modelLmfitMultiBasic(dic_lmfit['PK'][i].params, i, x_long)
    if options_dic['use_fit_doster']: 
        #if options_dic['doster_single_fit'] == True we have only on item in list -> set index 'il' of list to 0
        il = i if (options_dic['models']['Do']['doster_single_fit'] == False) else 0
        doster_y_fit = doster_modelLmfitMultiBasic(dic_lmfit['Do'][il].params, i, x_long)
    Yi_y_fit = yi_modelLmfitMultiBasic(dic_lmfit['Yi'][i].params, i, x_long)
    
    ft = fig11.add_subplot(2,1,nbplots)
    #ft .errorbar(used_q, used_data[i, :],yerr=used_data_err[i, :], marker='o',  linestyle='None',label='T = %iK' % used_T[i])
    ft .plot(used_q, used_data[i, :], marker='o',  linestyle='None',label='T = %iK' % used_T[i])
    ft .plot(x_long, np.exp(GA_y_fit), '-', label=r'GA  %.2f$\AA^{2}$' % (dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].value*-3))
    ft .plot(x_long, PK_y_fit, '-', label=r'PK  $\,$%.2f$\AA^{2}$' % (dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)].value**2*3))
    #il
    if options_dic['use_fit_doster']: 
        ft .plot(x_long, doster_y_fit, '-', label=r'DO  $\,$%.2f$\AA^{2}$' % (dic_lmfit['Do'][il].params['msdG_%02d' % (i+1)].value*3))
    ft .plot(x_long, Yi_y_fit, '-', label=r'Yi   $\,$%.2f$\AA^{2}$' % (dic_lmfit['Yi'][i].params['msd_%02d' % (i+1)].value*0.5))
    
    handles,labels = ft .get_legend_handles_labels()
    #handles = [handles[-1]]+handles[0:-1]
    #labels = [labels[-1]]+labels[0:-1]
    ft.legend(handles,labels,prop={'size':10},loc=1,labelspacing=0.4)
    ft.set_xlabel(r'Q [$\AA^{-1}$]')
    ft.set_ylabel(r'EISF ($\frac{I_{T}}{I_{20K}}}$)')
    plt.ylim((0,1.))
fig11.tight_layout()
'''
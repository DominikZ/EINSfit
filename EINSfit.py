'''This module is used to fit elastic incoherent neutron scatting (EINS) data. 
    The main class is EINSfit. Most of the other functions are used by this class and are usually not needed.
'''

__version__ = '0.9.0b1'

import numpy as np
import matplotlib
#matplotlib.use('Agg') #for plotting without display
import matplotlib.pyplot as plt
#from lmfit import minimize, Parameters
import lmfit

#for exit
import sys
import os
#for paths
from pathlib import Path

import copy
import errno #for creating directory

#for saving
import pickle
import json

#for savely loading
import configparser #for ini file
import ast #for save evaluation of read input: ast.literal_eval

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


#checks if numpy arrays or single items in dictionaris are the same
def check_dic_same(d1,d2,key=None):
    if key is not None:
        if isinstance(d1[key], np.ndarray):
            if not np.array_equal(d1[key],d2[key]):
                print(key)
                print('n')
        elif isinstance(d1[key], dict):
            for new_key in d1[key]:
                check_dic_same(d1[key],d2[key],new_key)
        else:
            if not d1[key]==d2[key]:
                print('n')
    else:
        for new_key in d1:
            check_dic_same(d1,d2,new_key)

#hack to load pickles from python2 when using python3:
def python3_load_pickle_of_python2(input):
    with open(input, 'rb') as infile:
        u = pickle._Unpickler(infile)
        u.encoding = 'latin1'
        output=u.load()
    return output

def create_dir(DirName):
    if not os.path.exists(DirName):
        try:
            os.makedirs(DirName)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def sumI_diffQ(Qaxis,valueQ,errorQ,increment):
    data=np.empty(0)
    error=np.empty(0)
    QaxisAvg=np.empty(0)
    for i in range(0,len(valueQ[:,0]),increment):
        if i<len(valueQ[:,0])-increment:
            end=i+increment
            div=increment
        else:
            end=len(valueQ[:,0])
            div=-i+len(valueQ[:,0])
        data=np.hstack([data,np.sum(valueQ[i:end,:],0)])
        errorSumSqrt=np.sqrt(np.sum(errorQ[i:end,:]**2,0))
        error=np.hstack([error,errorSumSqrt])
        QaxisAvg=np.hstack([QaxisAvg,np.sum(Qaxis[i:end],0)/div])
        #print len(data)
    data=np.reshape(data,(-1,len(valueQ[0,:])))
    error=np.reshape(error,(-1,len(valueQ[0,:])))
    
    #return data
    return QaxisAvg,data,error

#### functions used by class !!

def is_sorted(a):
    for idx,val in enumerate(a[:-1]):
         if a[idx+1] < val :
               return False
    return True

#to save file in json even if file uses np.ndarray data type -> np.array will be transformed to list
#use with: with open('data.json', 'w') as f:
#     json.dump(dictionary_file,f, indent=4, cls=NumpyEncoder)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#save data in json and load json data
def load_json(input,convert_lists_to_numpy=False):
    with open(input, 'r') as infile:
        output=json.load(infile)
    if convert_lists_to_numpy:
        convert_lists_to_numpy_in_dic(output)
    return output

def save_json(input,name_out): 
    with open(name_out, 'w') as outfile:
        json.dump(input, outfile, indent=4, cls=NumpyEncoder)

#save data in pickle and load pickled data
def load_pickle(input):
    with open(input, 'rb') as infile:
        output=pickle.load(infile)
    return output
    
def save_pickle(input,name_out): 
    with open(name_out, 'wb') as outfile:
        pickle.dump(input, outfile, protocol=2)#pickle.HIGHEST_PROTOCOL)

def convert_lists_to_numpy_in_dic(dic_in,key=None):
    if not isinstance(dic_in,dict):
        raise TypeError('Input variable \"dic_int\" has to be an dictionary.')
    if key is not None:
        if isinstance(dic_in[key], list):
            dic_in[key]=np.array(dic_in[key])
        elif isinstance(dic_in[key], dict):
            for new_key in dic_in[key]:
                convert_lists_to_numpy_in_dic(dic_in[key],new_key)
        else:
            pass
    else:
        for new_key in dic_in:
            convert_lists_to_numpy_in_dic(dic_in,new_key)

def save_data(fileName,data,plain=True, numpyFormat=True):
    if plain:
        if isinstance(data,dict):
            with open(Path(fileName).with_suffix('.txt'),'w') as f:
                f.write(str(data))
        elif isinstance(data,np.ndarray):
            np.savetxt(Path(fileName).with_suffix('.txt'),data)
        else:
            raise TypeError('Data of type \'%s\' can not be saved as plain text.'%type(data))
    if numpyFormat:
        np.save(fileName,data)
    return

def remove_data(fileName,plain=True, numpyFormat=True):
    #unlink() of Path deletes File, but gives Error if File does not exist
    if plain:
        try:
            Path(fileName).with_suffix('.txt').unlink()
        except FileNotFoundError:
            pass
    if numpyFormat:
        try:
            Path(fileName).with_suffix('.npy').unlink()
        except FileNotFoundError:
            pass
    return

def _print_diff_models(m1,m2, path=''):
    #instances=['np.ndarray','list','tuple']
    for k in m1:
        if k not in m2:
            print (path, ':')
            print (' ! \'%s\' as key not in m2'%k)
        else:
            if isinstance(m1[k],np.ndarray):
                if not np.array_equal(m1[k],m2[k]):
                    print ('%s[\'%s\']: %s != %s'%(path,k,m1[k],m2[k]))
            elif isinstance(m1[k],list):
                for idx,l in enumerate(m1[k]):
                    if not l==m2[k][idx]:
                        print('%s[\'%s\']: %s != %s'%(path,k,m1[k],m2[k]))
                        break
            elif isinstance(m1[k],tuple):
                if isinstance(m1[k][0],np.ndarray):
                    if not np.array_equal(m1[k][0],m2[k][0]):
                        print ('%s[\'%s\']: %s != %s'%(path,k,m1[k],m2[k]))
                else:
                    assert(0),'Something is not working ... sorry'
            else:
                #print(m1[k],type(m1[k]))
                if not m1[k]==m2[k]:
                    print('%s[\'%s\']: %s != %s'%(path,k,m1[k],m2[k]))

def print_diff_in_config_1D(d1, d2, path=''):
    ''' Prints the difference between d1 and d2:
        Only works/tested with options_dic and fitting_dic
        Does not take into account if d2 is bigger than d1
        For that see: print_diff_in_config
    '''
    for k in d1:
        if k not in d2:
            print (path, ':')
            print (' ! \'%s\' as key not in d2.'%k)
        elif k=='models':
            #pass
            for model in d1[k]:
                m1=d1[k][model]
                if model in d2[k]:
                    m2=d2[k][model]
                    _print_diff_models(m1,m2, path=path+'[\'%s\'][\'%s\']'%('models',model))
                else:
                    print(path, ':')
                    print (' ! Model \'%s\' not in d2.'%model)         
        else:
            if type(d1[k]) is dict:
                if path == '':
                    path_new = '[\'%s\']'%k
                else:
                    path_new = path + ' [\'%s\']'%k
                print_diff_in_config_1D(d1[k],d2[k], path_new)
            else:
                if isinstance(d1[k],np.ndarray):
                    if not np.array_equal(d1[k],d2[k]):
                        print (path, ':')
                        print (' d1: [\'%s\'] = %s'%(k, d1[k]) )
                        print (' d2: [\'%s\'] = %s'%(k, d2[k]) )
                elif d1[k] != d2[k]:
                    print (path, ':')
                    print (' d1: [\'%s\'] = %s'%(k, d1[k]) )
                    print (' d2: [\'%s\'] = %s'%(k, d2[k]) )

def readIN_elascan(fileBase,return_all=False):
    ''' Reads in elascan files '*_q.dat' and '*_t.dat' saved by LAMP.
        Returns the intensity data as 2D numpy.array with order [T,Q] and as 1D numpy.array T and Q:
            --> data[T,Q], data_err[T,Q], T, Q

        Parameters:
        -----------
        fileBase : str
            File base name of elascan '*_q.dat' and '*_t.dat' files
        return_all : bool, optional
            If True, returns also inversed matrix of Q values:
                --> data[T,Q], data_err[T,Q], inv_data[Q,T], inv_data_err[Q,T], T, Q
    '''
    rawDataQ=np.loadtxt(Path(str(fileBase)+'_q.dat'))
    rawDataT=np.loadtxt(Path(str(fileBase)+'_t.dat'))
    
    Q=rawDataQ[:,0]
    T=rawDataT[:,0]
    #print Q,T

    DataT_value=rawDataT[:,1::2]
    DataT_error=rawDataT[:,2::2]
    DataQ_value=rawDataQ[:,1::2]
    DataQ_error=rawDataQ[:,2::2]
    
    #check shape
    if not (Q.shape[0] == DataQ_value.shape[0] and T.shape[0] == DataT_value.shape[0] and 
            Q.shape[0] == DataT_value.shape[1] and T.shape[0] == DataQ_value.shape[1]):
        sys.exit('Abort: Array of values has not the same shape as Q and T or are interchanged: Check input files!!')
    #check if T values/errors and Q values/errors are the same
    if not np.array_equal(DataT_value.T,DataQ_value):
        sys.exit('ERROR: Values %s_q.dat and %s_t.dat file are not equivalent' % (fileBase,fileBase))
    if not np.array_equal(DataT_error.T,DataQ_error):
        sys.exit('ERROR: Errors %s_q.dat and %s_t.dat file are not equivalent' % (fileBase,fileBase))
    
    
    #return data
    if not return_all:
        #standard return
        return DataT_value,DataT_error,T,Q
    else:
        return DataT_value,DataT_error,DataQ_value,DataQ_error,T,Q

def write_or_delete_warning(filename,warnings):
    # if warnings is not None:
    #     if not isinstance(warnings,list):
    #         raise TypeError('Input variable \"warnings\" has to be of type \"list\".')
    if (warnings is None) or (len(warnings)==0):
        #unlink() of Path deletes File, but gives Error if File does not exist
        try:
            Path(filename).unlink()
        except FileNotFoundError:
            pass
    else:
        with open(filename,"w") as f:
            for warning in warnings:
                f.write('# %s\n'%warning)


def write_fit_report(filename,data,T,q_values=None):
    with open(filename,"w") as f:
        f.write('# Results of fit')
        if q_values is not None: f.write(' with model fit region: q=%.2f to q=%.2f' % (q_values[0],q_values[-1]))
        f.write('\n')
        if len(data) == 1:
            f.write(lmfit.fit_report(data[0]))
        else:
            for i in range(0,len(T),1):
                f.write('TemperatureSet %i with T=%.2f\n' % (i,T[i]))
                f.write(lmfit.fit_report(data[i]))
                f.write('\n-----------------------------------\n')
            
def write_fit_report2(filename,data,T,q_values=None):
    with open(filename,"w") as f:
        f.write('# Results of fit')
        if q_values is not None: f.write(' with model fit region: q=%.2f to q=%.2f' % (q_values[0],q_values[-1]))
        f.write('\n')
        string = 'T '
        for key in data[0].params.keys():
            string=string+key+' '+key+'-stderr '
        string=string+'nvarys ndata nfree '
        string=string+'chisqr redchi '
        string=string+'aic bic '
        f.write(string+'\n')
        string = ''
        for i in range(0,len(T),1):
            string = '%.2f ' % (T[i])
            for key in data[i].params.keys():
                string=string+'%e ' %(data[i].params[key].value)
                #print error, if not existent then print 0
                if data[i].params[key].stderr is not None:
                    string=string+'%e ' %(data[i].params[key].stderr)
                else:
                    string=string+'%e ' %(0)
            string=string+'%i %i %i ' %(data[i].nvarys,data[i].ndata,data[i].nfree)
            string=string+'%e %e ' %(data[i].chisqr,data[i].redchi)
            string=string+'%e %e ' %(data[i].aic,data[i].bic)
            f.write(string+'\n')
            
def write_fit_report3(filename,data,T,q_values=None):
    assert(len(data)==1) # only working for global fit at the moment!
    Tl= len(T)
    results=sorted(data[0].params.items())
    NrVariables=len(results)/Tl
    assert(NrVariables*Tl==len(results))
    space=' '
    summary=''
    fline='T'+space
    for idx,t in enumerate(T):
        line=str(t)+space
        for key,parameter in results [idx::Tl]: # pylint: disable=unused-variable
            if idx==0:
                fline += getattr(parameter,'name') + space + getattr(parameter,'name') + '_Err' + space
            line += '%e' % getattr(parameter,'value') + space + '%e' % getattr(parameter,'stderr') + space
        summary+=line + '\n'
    with open(filename,"w") as f:
        f.write('# Results of fit')
        if q_values is not None: f.write(' with model fit region: q=%.2f to q=%.2f' % (q_values[0],q_values[-1]))
        f.write('\n')
        f.write(fline+'\n'+summary)

## fitting models for lmfit
def pk_modelLmfitMultiBasic(params,i, x, data=None, eps=None, no_weighting=False):
    vals = params.valuesdict()
    offset = np.float64(vals['offset_%02d' % (i+1)])
    sigma = np.float64(vals['sigma_%02d' % (i+1)])
    beta = np.float64(vals['beta_%02d' % (i+1)])
    
    model = offset  / ( ( 1. + ( (sigma*x)**2. / beta ))**beta )
    
    if data is None:
        return model
    if eps is None or no_weighting:
        return (model-data)
    return (model - data)/eps
    
def doster_modelLmfitMultiBasic(params,i, x, data=None, eps=None, no_weighting=False):
    vals = params.valuesdict()
    offset = np.float64(vals['offset_%02d' % (i+1)])
    #G = np.float64(vals['G_%02d' % (i+1)])
    msdG = np.float64(vals['msdG_%02d' % (i+1)])
    p12 = np.float64(vals['p12_%02d' % (i+1)])
    #p2 = np.float64(vals['p2_%02d' % (i+1)])
    d = np.float64(vals['d_%02d' % (i+1)])
    
    model = offset  * np.exp(-x**2*msdG)*(1.-2.*p12*(1.-np.sinc(x*d/np.pi))) #attention! python sinc(x) uses sin(x*pi)/(x*pi)
    
    if data is None:
        return model
    if eps is None or no_weighting:
        return (model-data)
    return (model - data)/eps
    
def doster_modelLmfitMultiBasic_Multi(params,x_in, data, eps=None, no_weighting=False):
    
    ndata, nx = data.shape
    resid = 0.0*data[:]
    assert(len(x_in) == nx)
    # make residual per data set
    if eps is None or no_weighting:
        for i in range(ndata):
            resid[i, :] = data[i, :] - doster_modelLmfitMultiBasic(params,i, x_in)
        # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()
    else:
        for i in range(ndata):
            resid[i, :] = ( data[i, :] - doster_modelLmfitMultiBasic(params,i, x_in) ) / eps[i, :]
        # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()
    
def yi_modelLmfitMultiBasic(params,i, x, data=None, eps=None, no_weighting=False):
    vals = params.valuesdict()
    offset = np.float64(vals['offset_%02d' % (i+1)])
    msd = np.float64(vals['msd_%02d' % (i+1)])
    sigma = np.float64(vals['sigma_%02d' % (i+1)])

    model = offset * np.exp(-1/6.*(x**2)*msd)*(1+(x**4)/72.*sigma**2)
    
    if data is None:
        return model
    if eps is None or no_weighting:
        return (model-data)
    return (model - data)/eps
    
def twoExp_modelLmfitMultiBasic(params,i, x, data=None, eps=None, no_weighting=False):
    vals = params.valuesdict()
    msd1 = np.float64(vals['msd1_%02d' % (i+1)])
    msd2 = np.float64(vals['msd2_%02d' % (i+1)])
    p1=np.float64(vals['p1_%02d' % (i+1)])
    offset = np.float64(vals['offset_%02d' % (i+1)])

    model = (p1*np.exp(-x**2*msd1)  + (1.-p1)*np.exp(-x**2*msd2)) * offset
    if data is None:
        return model
    if eps is None or no_weighting:
        return (model-data)
    return (model - data)/eps

def lin_modelLmfitMultiBasic(params,i, x, data=None, eps=None, no_weighting=False):
    vals = params.valuesdict()
    slope = np.float64(vals['slope_%02d' % (i+1)])
    intercept = np.float64(vals['intercept_%02d' % (i+1)])

    model = slope * x + intercept
    
    if data is None:
        return model
    if eps is None or no_weighting:
        return (model-data)
    return (model - data)/eps
    
class EINSfit:
    ''' Fits different EINS models to one data set of multiple temperatures scans.
        Data set has to be defined via a Path (string) datafile='your_elascan_baseName' or 'your_save_directory'.
        The datafile has to be 
        - the prefix(='your_elascan_BaseName') of the two elascan output files from LAMP (prefix+'_q.dat' and prefix+'_t.dat') 
        or
        - the directory of your previously saved data set. 
            --> This will ONLY load the raw input data + used input dictionary 
            --> This will NOT load fit results or the configuration file (=config_dic), if wanted, load saved config file with read_config_file()
        or
        - data saved in a dictionary with entries 'raw_data','raw_data_err','raw_q','raw_T'
            --> data has to be a numpy array: numpy.ndarray
            --> 'raw_q' and 'raw_T' are 1D arrays
            --> 'raw_data' and 'raw_data_err' are 2D arrays, axis1=len(raw_T) and axis2=len(raw_q)

        Parameters
        ----------
        datafile : string or dict, mandatory
            "data_type" == 'elascan'  : string = 'your_elascan_BaseName' (without '_q.dat' or '_t.dat')
            "data_type" == 'save'     : string = 'your_save_BaseDirectory'
            "data_type" == 'numpy_dic': dict   = {'raw_data': np.ndarray[q,T],'raw_data_err': np.ndarray[q,T],'raw_T': np.ndarray,'raw_q': np.ndarray}

        name : string, optional if not "data_type" = 'save'
            Name you want to give your data set. 
            ! Must be set if "data_type" = 'save'

        data_type : 'elascan' or 'save' or 'numpy_dic', optional
            Defines your data input type.
            'elascan'   = load elascan output files from LAMP
            'save'      = load directory of your previously saved data set
            'numpy_dic' = load data dictionary which has to be defined in the input variable 'datafile'
        
        dic_data_to_use : {'T_start': int, 'T_end': int, 'Q_min': int, 'Q_max': int, 'delete_specific_Q-values_list': [] } , optional
            Dictionary which defines the used data from the loaded data set.
            All values are optional, if set to None or not defined all values are used.
            'T_start' : number of first used temperature value (type: int)
            'T_end' : number of last used temperature value (type: int)
            'Q_min' : number of first used q value (type: int)
            'Q_max' : number of last used q value (type: int)
            'delete_specific_Q-values_list' : list of q values which should be excluded, has to be the exact value! (type list)

        save_dir_path : string, optional
            Defines where you want to save your data (Base directory).

        Attributes
        ----------
        name : string, name of your data set

        Readable Attributes (only a copy of the original variable is returned)
        --------------------
        config_dic : dict, dictionary of config for data fitting,
            To change this dictionary, use set_config_file() or read_config_file()
            For a nice overview over this dictionary, use print_config()

        raw_data_type : string, return the loaded data type ('elascan' or 'save')
        raw_file_path : string, return loaded data set path

        raw_T : numpy.ndarray, return raw temperature data
        raw_q : numpy.ndarray, return raw Q data
        raw_data : numpy.ndarray, return raw EISF data as 2D numpy.array with [T,Q]
        raw_data_err : numpy.ndarray, return raw EISF data error as 2D numpy.array with [T,Q]
        
        used_T : numpy.ndarray, return used temperature data
        used_q : numpy.ndarray, return used Q data
        used_data : numpy.ndarray, return used EISF data as 2D numpy.array with [T,Q]
        used_data_err : numpy.ndarray, return used EISF data error as 2D numpy.array with [T,Q]
        used_data_log : numpy.ndarray, return used log(EISF data) as 2D numpy.array with [T,Q]
        used_data_err_log : numpy.ndarray, return used log(EISF data) error as 2D numpy.array with [T,Q]
        used_Qmin : int, return used first Q value in comparision to raw data
        used_Qmax : int, return used last Q value in comparision to raw data
    '''

    def __init__(self,datafile,name=None,data_type='elascan',dic_data_to_use=None,save_dir_path=None):
        ''' Initializes class object, for help see help(EINS_fit)'''
        allowed_data_types=['elascan','save','numpy_dic']#'sim']
        if not data_type in allowed_data_types:
            raise ValueError('\"data_type\" has to be \'%s\''%allowed_data_types)
        if data_type != 'numpy_dic':
            file_path=Path(datafile)
        #assert(file_path.exists()), 'Data path %s does not exist.'%file_path 
        
        if data_type=='elascan':
            raw_DataT_value,raw_DataT_error,raw_T,raw_q=readIN_elascan(file_path)
        elif data_type=='save':
            if name is None:
                raise ValueError('Input parameter \"name\" has to be set for \"data_type\"=\'save\'!')
            created_data_is_different_from_loaded_data_file=False # if newly created object is different from saved files, make sure a new folder is created if data is saved!
            file_path_save=file_path/('python_data_' + str(name))
            if file_path_save.exists():
                raw_DataT_value=np.load(file_path_save/'raw_data.npy')
                raw_DataT_error=np.load(file_path_save/'raw_data_err.npy')
                raw_T=np.load(file_path_save/'raw_T.npy')
                raw_q=np.load(file_path_save/'raw_q.npy')
                file_tmp=file_path_save/'dic_data_to_use.txt'
            else:
                raise OSError('File path \'%s\' does not exist!'%file_path_save)
            if dic_data_to_use is None:
                #if there is no given dic_data then load saved one IF it exists
                if file_tmp.is_file():
                    with open(file_path_save/'dic_data_to_use.txt','r') as f:
                        dic_data_to_use = ast.literal_eval(f.read())
                    print('Saved \'dic_data_to_use.txt\' is used to transform raw Data to used Data for fits!')
                    print('The loaded config is:')
                    for key in dic_data_to_use:
                        print('[\'%s\'] = %s'%(key,dic_data_to_use[key]))
                else:
                    pass
            else:
                created_data_is_different_from_loaded_data_file=True
                #if there is a given dic_data then print a warning that the load data set is different from what you are doing!
                if file_tmp.is_file():
                    print('Warning: Existing dictionary file \'%s\' was not used!'%(file_tmp,) )
                else:
                    print('Warning: Dictionary file \'%s\' does NOT exist! Therefore, raw data = used data in loaded data set.'%(file_tmp,) )
                print('Instead following given options are used:')
                for key in dic_data_to_use:
                        print('[\'%s\'] = %s'%(key,dic_data_to_use[key]))
            print('-'*40)
        elif data_type=='numpy_dic':
            file_path=None
            if name is None:
                name='input_data-from-numpy_dic'
            if not isinstance(datafile,dict):
                raise ValueError('Input variable \"datafile\" has to be a dictionary.')
            allowed_data_names=['raw_data','raw_data_err','raw_T','raw_q']
            for key in datafile:
                if key in allowed_data_names:
                    if not isinstance(datafile[key],np.ndarray):
                        raise TypeError('Entry datafile[\'%s\'] has to be a numpy array.'%key)
                else:
                    raise KeyError('Key = \'%s\' not allowed in datafile dictionary.'%key)
                
            raw_DataT_value=datafile['raw_data']
            raw_DataT_error=datafile['raw_data_err']
            raw_T=datafile['raw_T']
            raw_q=datafile['raw_q']
            if not (raw_q.shape[0] == raw_DataT_value.shape[1] and raw_T.shape[0] == raw_DataT_value.shape[0]):
                raise  ValueError('raw_data[T,q] has not the same shape as raw_q and raw_T.')
            if not (raw_DataT_value.shape == raw_DataT_error.shape):
                raise  ValueError('raw_data has not the same shape as raw_data_err.')
        else:
            raise ValueError('Data type for \"data_type\" = \'%s\' not supported.'%data_type)
        
        #save raw data read:
        self._raw_data_type=data_type
        self._raw_file_path=file_path
        self._raw_q=raw_q
        self._raw_data=raw_DataT_value
        self._raw_data_err=raw_DataT_error
        self._raw_T=raw_T
        self._dic_transform_rawData_to_usedData=copy.deepcopy(dic_data_to_use)
        self._raw_exists=True
        
        #convert raw data to data used
        self._used_exists=False
        self._define_data_used(self._dic_transform_rawData_to_usedData)
        
        #define name of sample
        self._name = str(name) if (name is not None) else file_path.name

        
        #define basic options and save them to self._config_dic
        self._config_dic=None
        self._define_default_config()
        
        #save location, create directory
        #save location is given
        if save_dir_path is not None:
            save_dir_path=self._prepare_save_dir(Path(save_dir_path),no_sub_folder=True)
        #save location is not given
        else:
            if data_type != 'save':
                save_dir_path=self._prepare_save_dir(Path(os.getcwd()))
            else:
                tmp_bool=created_data_is_different_from_loaded_data_file
                save_dir_path=self._prepare_save_dir(file_path,no_sub_folder=not tmp_bool,use_empty_folder=tmp_bool)
        #change to dir to get full path
        pwd=os.getcwd()
        os.chdir(save_dir_path)
        #save full path to instance variable
        self._save_dir_path=Path(os.getcwd())
        os.chdir(pwd) #change back to old dir
        #print where data is saved
        print('Data will be saved in folder: \'%s\''%(self._save_dir_path))
        
        #logging of warnings and process
        self._warnings_fitting_options=None
        self._warnings_fitting_process=None 
        #self._log_fitting_process=None
        
        #initialize record lists
        self._record_nb=0
        self._record_config_dic = []
        self._record_fitting_options_dic = []
        self._record_warnings_fitting_options = []
        self._record_warnings_fitting_process = []
        self._record_results_lmfit_dic = []
        self._record_nice_results_dic = []
    
    def __del__(self):
        ''' Remove created dictionary if it is empty.'''
        #remove created directory if empty
        try:
            os.rmdir(self._save_dir_path)
        except (OSError,AttributeError):
            #do not raise Error if Dir was not empty or self._save_dir_path was not defined
            pass

    @staticmethod
    def _dic_use_value_or_default(dic,_key,_default):
        assert(isinstance(dic, dict))
        if _key in dic:
            val=dic[_key]
            if val:
                if type(_default)!=type(val):
                    raise TypeError('Value passed has not the type needed. Value has to be of type \'%s\''%type(_default))
                return val
            else:
                return _default
        else:
            return _default
        
    def _define_data_used(self,dic_data_to_use={}):
        assert(not self._used_exists),'Used data already exists'
        if dic_data_to_use is None:
            dic_data_to_use={}
        ts=self._dic_use_value_or_default(dic_data_to_use,'T_start',int(0))
        tf=self._dic_use_value_or_default(dic_data_to_use,'T_end',len(self._raw_T))
        self._used_T=self._raw_T[ts:tf]
        self._used_Qmin=self._dic_use_value_or_default(dic_data_to_use,'Q_min',int(0))
        self._used_Qmax=self._dic_use_value_or_default(dic_data_to_use,'Q_max',len(self._raw_q))
        self._used_q=self._raw_q[self._used_Qmin:self._used_Qmax]
        self._used_data=self._raw_data[ts:tf,self._used_Qmin:self._used_Qmax]
        self._used_data_err=self._raw_data_err[ts:tf,self._used_Qmin:self._used_Qmax]
        
        dic_q_del=self._dic_use_value_or_default(dic_data_to_use,'delete_specific_Q-values_list',[])
        #delete specific Qs defined in options_dix['delete_specific_Q-values_list']:
        for q_del in dic_q_del:
            take_values_idx=np.where(np.invert(np.isclose(q_del,self._used_q)))
            self._used_q=self._used_q[take_values_idx]
            self._used_data=np.take(self._used_data,take_values_idx[0],axis=1)
            self._used_data_err=np.take(self._used_data_err,take_values_idx[0],axis=1)
            
        #check for data consisitency
        if self._used_data.min()<0:
            raise ValueError('Program stopped since there exists an intensity value smaller than 0: %f (for T=%i,Q=%.5f)' %(self._used_data.min(), self._used_T[self._used_data.min(1).argmin()], self._used_q[self._used_data.min(0).argmin()]) )
        
        #sorting
        if not is_sorted(self._used_T):
            sort_arr=self._used_T.argsort()
            self._used_T=self._used_T[sort_arr]
            self._used_data=self._used_data[sort_arr,:]
            self._used_data_err=self._used_data_err[sort_arr,:]
            print('Info: Used data was sorted in temperature.')
        if not is_sorted(self._used_q):
            sort_arr=self._used_q.argsort()
            self._used_q=self._used_q[sort_arr]
            self._used_data=self._used_data[:,sort_arr]
            self._used_data_err=self._used_data_err[:,sort_arr]
            print('Info: Used data was sorted in Q.')

        self._used_data_log=np.log(self._used_data)
        self._used_data_err_log=self._used_data_err/self._used_data
        
        self._used_exists=True

        #info
        if (self._used_q.shape != self._raw_q.shape) or (self._used_T.shape != self._raw_T.shape):
            print('Info: Used data is NOT the same as Raw data read.')
        else:
            print('Info: Used data is the same as Raw data read.')
        print('Used Q values: %s'%self._used_q)
        print('Used T values: %s'%self._used_T)
        print('-'*40 + '\n')
        
        return
    
    def _define_default_config(self):
        config_dic={}
        config_dic['Global']={}
        config_dic['Global']['GA_refit_if_interceptGreater0_OR_slopeGreater0']=False
        config_dic['Global']['GA_intercept_max']=0.4
        
        #offset settings, only taken into account if offset = false, 
        #   but max should be equal or bigger than exp(options_dic['GA_intercept_max'])
        config_dic['Global']['offset_start_min_max']=[0.9, 0.3, np.exp(config_dic['Global']['GA_intercept_max'])]
        config_dic['Global']['fix_to_offset'] = True
        config_dic['Global']['no_weighting_in_fit'] = False    
        config_dic['Global']['print_report']=False
        #plot settings
        config_dic['Global']['plots_fitsAllT_Qrange_min-max']=[0,4.8]
        
        config_dic['GA']={}
        config_dic['GA']['Q_max']=np.sqrt(4)
        config_dic['PK']={}
        config_dic['PK']['Q_max']=4.5
        config_dic['PK']['sigma_start-min-max']=[1.0, 0.01, 10.]
        config_dic['PK']['beta_start-min-max']=[0.5, 0.01, 100.]
        config_dic['Yi']={}
        config_dic['Yi']['Q_max']=4.5
        config_dic['Yi']['sigma_start-min-max']=[0.6, 1e-7, 5.]
        config_dic['Yi']['msd_start-min-max']=[0.2, 1e-7, 5.]
        config_dic['Do']={}
        config_dic['Do']['use_fit_doster']=False
        config_dic['Do']['doster_d_fixed']=False
        config_dic['Do']['doster_d_val']=1.5
        
        self._config_dic=config_dic

        return

    def set_config_file(self,dic):
        ''' Set one or more values to config file via a nested dictionary.
        '''
        if not isinstance(dic,dict):
            raise TypeError('\"dic\" has to be a nested dictionary (section --> key = value), eg: {\'section\' : {\'key\' : value}}')
        ''' ATTENTION:
            Attention, if you load self.config_file, only a copy of the original self._config_file is given!
              !!  --> In order to change or add a value to the original config file, you have to use self._config_file !!
        '''
        config_dic=self.config_dic
        set_sections=[]
        for section in config_dic:
            if section in dic:
                set_sections.append(section)
        if len(set_sections) > 0:
            for section in set_sections:
                if not isinstance(dic[section],dict):
                    raise TypeError('\"dic\" has to be a nested dictionary (section --> key = value), eg: {\'section\' : {\'key\' : value}}')
                print('-'*24,'\nReading given settings and following values are updated:')
                for key in dic[section]:
                    if key in config_dic[section]:
                        if isinstance(dic[section][key],str):
                            #to change value, you have to use self._config_dic, see ATTENTION comment above
                            self._config_dic[section][key]=ast.literal_eval(dic[section][key])
                        else:
                            #to change value, you have to use self._config_dic, see ATTENTION comment above
                            self._config_dic[section][key]=dic[section][key]
                        print ('[\'%s\'][\'%s\'] = %s'%(section,key,self.config_dic[section][key]))
        else:
            raise ValueError('Given dictionary \"dic\" has no allowed section, allowed are: %s'%config_dic.keys())
        
        print('Reading done.','-'*10)

    def read_config_file(self,filename=None,):
        '''Reads the config from given file and overwrite basic_options dictionary with new Values.
            Parameters:
            -----------
            filename : string
                Config file location.
        '''
        if filename is None:
            filename = self._save_dir_path / Path('config_file_%s.ini' %self.name)
        else:
            filename = Path(filename)
        config = configparser.ConfigParser()
        config.optionxform = str #read lower and upper case letters, if not it transforms all letters to lowercase letters
        if filename.is_file():
            config.read(filename)
        else:
            raise OSError('File \"%s\" not found!'%filename)
        
        print ('Reading config file \'%s\' and following data was read successfully:'%(filename))

        ''' ATTENTION:
            Attention, if you load self.config_file, only a copy of the original self._config_file is given!
              !!  --> In order to change or add a value to the original config file, you have to use self._config_file !!
        '''
        config_dic=self.config_dic
        for section in config:
            if section not in config_dic:
                continue
            for key in config[section]:
                if key in config_dic[section]:
                    #to change value, you have to use self._config_dic, see ATTENTION comment above
                    self._config_dic[section][key]=ast.literal_eval(config[section][key])
                    print ('[\'%s\'][\'%s\'] = %s'%(section,key,self.config_dic[section][key]))

    def save_config_file(self,record_nb=None,save_path=None,silent=False):
        '''Saves the config file (=basic_options dictionary) to a file.
            Parameters:
            -----------
            record_nb : int, optional
                Define from which record number you want to have the config (-1 = last record).
            save_path : string, optional
                Directory where config file is saved (save_path / 'config_file_SAMPLENAME.ini'). If None, the default save directory (can be changed with set_save_dir() ) is used.
            silent : bool, optional
                If True, no output is printed to the terminal.
        '''
        #get config file, if record_nb=None then take current config file
        dic=None
        warnings=None
        if record_nb is None:
            dic=self.config_dic
        else:
            #check if record_nb is valid
            self._check_record_nb(record_nb,silent=silent)
            dic=self._record_config_dic[record_nb]
            warnings=self._record_warnings_fitting_options[record_nb]

        #save result in str for export
        result_str=''
        key_order=self._sort_basic_options(dic)
        for section in key_order:
            result_str=result_str+'[%s]\n'%(section)
            for key in sorted(dic[section]):
                result_str=result_str+'%s=%s\n'%(key,dic[section][key])
        warnings_str=''
        if warnings is not None:
            warnings_str=warnings_str+'\n'
            warnings_str=warnings_str+'#ATTENTION, there have been warnings with this data set and basic options\n'
            for w in warnings:
                warnings_str=warnings_str+'# %s\n'%w
        total_str=result_str+warnings_str

        #get save location Path
        if save_path is None:
            save_location=self._save_dir_path
        else:
            save_location=self._prepare_save_dir(Path(save_path),no_sub_folder=True)
        save=save_location / ('config_file_%s.ini' %self.name)
        with open(save,'w') as f:
            f.write(total_str)
        if not silent:
            print('Config file successfully saved in %s.'%save)
        return
        
    @staticmethod
    def _prepare_save_dir(path,no_sub_folder=False,use_empty_folder=False):
        if not use_empty_folder: #standard behaviour
            i=1
            if no_sub_folder:
                save_location=Path(path)
            else:
                save_location=Path(path) / Path('save_%02d' %i)
            if not os.path.exists(save_location):
                    try:
                        os.makedirs(save_location)
                    except OSError as error:
                        if error.errno != errno.EEXIST:
                            raise
        else:
            #creating new empty main folder
            if no_sub_folder:
                raise ValueError('Option not supported with use_empty_folder = True')
            for i in range(0,100,1):
                save_location=Path(path) / Path('save_%02d' %i)
                if not os.path.exists(save_location):
                    try:
                        os.makedirs(save_location)
                    except OSError as error:
                        if error.errno != errno.EEXIST:
                            raise
                    break
        return save_location
    
    def _check_record_nb(self,record_nb,silent=False):
        if not isinstance(record_nb,int):
            raise TypeError('\"record_nb\" has to be an integer value!')

        if self._record_nb == 0:
            raise ValueError('No record saved until now. Run "run_fit()" method to create a record!')

        if record_nb<self._record_nb:
            str_last=''
            if record_nb==-1:
                str_last=' (= last record with number %i)'%(self._record_nb-1)
            if not silent:
                print("Return results of record number %i%s."%(record_nb,str_last))
        else:
            raise ValueError('Record number %i does not exist, try -1 for last entry' %record_nb)

    def set_save_dir(self,save_dir_path):
        '''Sets a new Base directory where data is saved as default.
            Parameters
            ----------
            save_dir_path : string
                Defines where you want to save your data (Base directory).
        '''
        
        old_save_dir_path=self._save_dir_path
        save_dir_path=self._prepare_save_dir(Path(save_dir_path),no_sub_folder=True)
        #get full dir and save to instance variable
        pwd=os.getcwd()
        os.chdir(save_dir_path)
        self._save_dir_path=Path(os.getcwd())
        os.chdir(pwd) #change back to old dir
        print('Data will be saved in folder: \'%s\''%(self._save_dir_path))
        
        #remove old directory if empty
        try:
            os.rmdir(old_save_dir_path)
        except OSError:
            pass

    def get_nice_results_dic(self,record_nb=-1,silent=False) -> dict:
        '''Returns nice dictionary with results for given record_nb.
            Parameters
            ----------
            record_nb : int, optional
                Define from which record number you want to have the results (-1 = last record).
            silent : bool, optional
                If True, no output is printed to the terminal.
        '''
        self._check_record_nb(record_nb,silent=silent)

        return copy.deepcopy(self._record_nice_results_dic[record_nb])


    def _create_nice_results_dic(self,record_nb=-1,silent=False) -> dict:
        '''Creates nice dictionary with results for given record_nb.
            Parameters
            ----------
            record_nb : int, optional
                Define from which record number you want to have the results (-1 = last record).
            silent : bool, optional
                If True, no output is printed to the terminal.
        '''
        self._check_record_nb(record_nb,silent=silent)
        dic_out={}
        dic=dic_out
        
        dic['name']=self.name

        dic['EISF_T']=self._used_T
        dic['EISF_Q']=self._used_q
        dic['EISF_data']=self._used_data
        dic['EISF_data_err']=self._used_data_err
        dic['EISF_data_log']=self._used_data_log
        dic['EISF_data_err_log']=self._used_data_err_log

        dic_out['raw_data']={}
        dic=dic_out['raw_data']
        dic['EISF_T']=self._raw_T
        dic['EISF_Q']=self._raw_q
        dic['EISF_data']=self._raw_data
        dic['EISF_data_err']=self._raw_data_err

        
        dic_out['Q_range']={}
        dic=dic_out['Q_range']
        dic_in=self._record_fitting_options_dic[record_nb]['models']
        for k_model in dic_in:
            dic[k_model]={}
            dic[k_model]['Q_fit']=dic_in[k_model]['qValues']
            dic[k_model]['Q_min']=dic[k_model]['Q_fit'][0]
            dic[k_model]['Q_max']=dic[k_model]['Q_fit'][-1]

        #write warnings if they exist
        warnings=self._record_warnings_fitting_process[record_nb]
        key='Warnings_during_fitting'
        if warnings is not None:
            dic_out[key]={}
            for w_idx,w in enumerate(warnings):
                dic_out[key]['%02d'%w_idx]=w
            print('Warning: There have been warnings during the fitting process! Created key = \'%s\' in nice results dictionary.' %key)
            if not silent:
                for w in warnings:
                    print(w)

        #write warnings if they exist
        warnings=self._record_warnings_fitting_options[record_nb]
        key='Warnings_fitting_options'
        if warnings is not None:
            dic_out[key]={}
            for w_idx,w in enumerate(warnings):
                dic_out[key]['%02d'%w_idx]=w
            print('Warning: There have been warnings creating the fitting options! Created key = \'%s\' in nice results dictionary.'%key)
            if not silent:
                for w in warnings:
                    print(w)


        nb_T=len(self._used_T)
        dic_lmfit=self._record_results_lmfit_dic[record_nb]

        #get MSD with factor 3 convention
        dic_out['MSD3']={}
        dic=dic_out['MSD3']
        for k_model in dic_lmfit:
            dic[k_model]={}
            dic[k_model]['vals']=np.zeros(nb_T)
            dic[k_model]['errors']=np.zeros(nb_T)
            for i in range(nb_T):
                dic[k_model]['vals'][i] = dic_lmfit[k_model][i].params['MSD_F3_calc_%02d'%(i+1)].value
                dic[k_model]['errors'][i] = dic_lmfit[k_model][i].params['MSD_F3_calc_%02d'%(i+1)].stderr

        #get STD with factor 3 convention
        dic_out['STD3']={}
        dic=dic_out['STD3']
        model_list=['PK','Yi'] #only add entries for models in this list
        for k_model in dic_lmfit:
            if k_model in model_list:
                dic[k_model]={}
                dic[k_model]['vals']=np.zeros(nb_T)
                dic[k_model]['errors']=np.zeros(nb_T)
                for i in range(nb_T):
                    dic[k_model]['vals'][i] = dic_lmfit[k_model][i].params['STD_F3_calc_%02d'%(i+1)].value
                    dic[k_model]['errors'][i] = dic_lmfit[k_model][i].params['STD_F3_calc_%02d'%(i+1)].stderr

        #get redchi of fit
        dic_out['redchi']={}
        dic=dic_out['redchi']
        for k_model in dic_lmfit:
            dic[k_model]=np.zeros(nb_T)
            for i in range(nb_T):
                dic[k_model][i] = dic_lmfit[k_model][i].redchi

        #get EISF(Q=0) for all models (same as GA if other models are fixed)
        dic_out['EISF(Q=0)']={}
        dic=dic_out['EISF(Q=0)']
        model_list=['PK','Yi','Do'] #models which have offset variable
        for k_model in dic_lmfit:
                dic[k_model]={}
                dic[k_model]['vals']=np.zeros(nb_T)
                dic[k_model]['errors']=np.zeros(nb_T)
                for i in range(nb_T):
                    if k_model == 'GA':
                        dic[k_model]['vals'][i] = np.exp(dic_lmfit[k_model][i].params['intercept_%02d' % (i+1)].value)
                        dic[k_model]['errors'][i] = dic[k_model]['vals'][i]*dic_lmfit[k_model][i].params['intercept_%02d' % (i+1)].stderr
                    elif k_model in model_list:
                        dic[k_model]['vals'][i] = dic_lmfit[k_model][i].params['offset_%02d' % (i+1)].value
                        dic[k_model]['errors'][i] = dic_lmfit[k_model][i].params['offset_%02d' % (i+1)].stderr

        #different sorting but same info as above
        dic_out2={}
        dic_in=self._record_fitting_options_dic[record_nb]['models']
        values_list=['Q_range','MSD3','STD3','redchi','EISF(Q=0)'] #values which will be copied in different order
        for k_model in dic_in:
            dic_out2[k_model]={}
            for k_value in values_list:
                if k_model in dic_out[k_value]:
                    if k_value not in ['Q_range',]:
                        dic_out2[k_model][k_value]=dic_out[k_value][k_model]
                    else:
                        for k_intern in dic_out[k_value][k_model]:
                            dic_out2[k_model][k_intern]=dic_out[k_value][k_model][k_intern]
        
        dic_out['models']=dic_out2

        return dic_out

    def give_fit_value(self,x,t=0,model='GA',record_nb=-1,GA_lin=False):
        '''Returns the y [=EISF(q)] value(s) to given x [=q] value(s) of requested model.
            Parameters:
            -----------
            x : float / array (or list) of floats
            t : int
                Number of temperature set (0=first, len(self._used_T)=last)
            model : 'GA' or 'PK' or 'Yi' or 'Do' or 'linAllQ', optional
                Name of desired model.
            record_nb : int, optional
                Define from which record number you want to have the results. (-1 = last record).
            GA_lin: bool, optional
                If True, function gives values of linear fit defined via ln(EISF) vs Q**2, e.g. for such a plot:
                    ln(EISF(Q))=Q**2 * MSD + log(EISF(0))
                     --> ln(EISF(x)) = give_fit_value(x=x**2, GA_lin=True)
                If False, definition as for other models:
                    EISF(Q)=exp(- Q**2 * MSD + EISF(0))
                     --> EISF(x) = give_fit_value(x=x, GA_lin=False)  (since internally x is squared)
        '''
        #returns y value using model=['GA','PK','Yi','Do','linAllQ'] and record_nb
        #checks
        self._check_record_nb(record_nb,silent=True)
        allowed_model_types=['GA','PK','Yi','Do','linAllQ']
        if not model in allowed_model_types:
            raise ValueError('model has to be %s'%allowed_model_types)
        if not isinstance(GA_lin,bool):
            raise ValueError('GA_lin has to be boolean (True or False) but is of type %s'%(type(GA_lin)) )
        if GA_lin:
            allowed_models_GA_lin=['GA','linAllQ']
            if model not in allowed_models_GA_lin:
                raise ValueError('GA_lin has to be False if not one of the following models is used: %s'%allowed_models_GA_lin)
    
        y=None
        i=t
        dic_lmfit=self._record_results_lmfit_dic[record_nb]
        options_dic=self._record_fitting_options_dic[record_nb]
        if model == 'GA':
            if GA_lin == False:
                y = np.exp(lin_modelLmfitMultiBasic(dic_lmfit[model][i].params, i, x*x))
            else:
                y = lin_modelLmfitMultiBasic(dic_lmfit[model][i].params, i, x)
        elif model == 'PK':
            y = pk_modelLmfitMultiBasic(dic_lmfit[model][i].params, i, x)
        elif model == 'Yi':
            y = yi_modelLmfitMultiBasic(dic_lmfit[model][i].params, i, x)
        else:
            if options_dic['use_fit_doster']:
                if model == 'Do': 
                    #if options_dic['doster_single_fit'] == True we have only on item in list -> set index 'il' of list to 0
                    il = i if (options_dic['models']['Do']['doster_single_fit'] == False) else 0
                    y = doster_modelLmfitMultiBasic(dic_lmfit[model][il].params, i, x)
                elif model == 'linAllQ':
                    if GA_lin == False:
                        y = np.exp(lin_modelLmfitMultiBasic(dic_lmfit[model][i].params, i, x*x))
                    else:
                        y = lin_modelLmfitMultiBasic(dic_lmfit[model][i].params, i, x)
            else:
                raise ValueError('Doster model [\'Do\'] was not fitted. No entry available ...')
        return y

    def _define_fitting_regions(self):
        ''' Define fitting_options dic from self.config_dic
        '''
        #intercept for GA fit options
        config_dic=self.config_dic
        warnings_list=[]
        options_dic={}
        options_dic['GA_refit_if_interceptGreater0_OR_slopeGreater0']=config_dic['Global']['GA_refit_if_interceptGreater0_OR_slopeGreater0']
        options_dic['GA_intercept_max']=config_dic['Global']['GA_intercept_max'] # exp(0.2)=1.22
        
        #offset settings, only taken into account if offset = false, 
        #   but max should be equal or bigger than exp(options_dic['GA_intercept_max'])
        options_dic['offset_start_min_max']=np.array(config_dic['Global']['offset_start_min_max'])
        
        x=self._used_q
        options_dic['models']={}
        #local Q-values for different models
        #GA model (linear fit):
        options_dic['models']['GA']={}
        dic=options_dic['models']['GA']

        dic['qValues_idx']=np.where( x<=config_dic['GA']['Q_max'] ) #take all values under value indicated
        dic['qValues']=x[dic['qValues_idx']]

        dic['qValues_squared']=dic['qValues']**2 #Q-values for linear fit of Q squared (x-axes)
        ##################
        #PK options START
        #####
        #PK model - q-values:
        options_dic['models']['PK']={}
        dic=options_dic['models']['PK']

        dic['qValues_idx']=np.where( (x<=config_dic['PK']['Q_max']) ) #take all values under value indicated
        dic['qValues']=x[dic['qValues_idx']]

        #starting fit parameters, sigma and beta
        dic['PK_sigma_start-min-max']=np.array(config_dic['PK']['sigma_start-min-max']) #np.array([1.0, 0.01, 10.])
        dic['PK_beta_start-min-max']=np.array(config_dic['PK']['beta_start-min-max']) #np.array([0.5, 0.01, 100.])
        
        #PK options END
        ##################
        
        ##################
        #Yi options START
        options_dic['models']['Yi']={}
        dic=options_dic['models']['Yi']

        dic['qValues_idx']=np.where(x<=config_dic['Yi']['Q_max']) #take all values under value indicated
        dic['qValues']=x[dic['qValues_idx']]


        #starting fit parameters, sigma (=std) and msd
        dic['Yi_sigma_start-min-max']=np.array(config_dic['Yi']['sigma_start-min-max']) #np.array([0.6, 1e-7, 5.])
        dic['Yi_msd_start-min-max']=np.array(config_dic['Yi']['msd_start-min-max']) #np.array([0.2, 1e-7, 5.])
        
        #Yi options END
        ##################
        
        ##################
        #doster options START
        options_dic['use_fit_doster']=config_dic['Do']['use_fit_doster'] #false
        if options_dic['use_fit_doster']:
            options_dic['models']['Do']={}
            dic=options_dic['models']['Do']
            
            dic['qValues_idx'] = copy.deepcopy(options_dic['models']['PK']['qValues_idx']) #doster_x_val=where operator: to get values of temperature i use: x[doster_x_val[i]]
            dic['qValues'] = copy.deepcopy(options_dic['models']['PK']['qValues']) #doster_x = Q-values
            
            #doster options configurable
            dic['doster_d_fixed'] = config_dic['Do']['doster_d_fixed']
            if dic['doster_d_fixed'] :
                dic['doster_d_val'] = config_dic['Do']['doster_d_val']

            #doster options only implemented here in function, not available in config dic
            dic['doster_single_fit'] = False #if True, gloabl fit is done
            dic['doster_lin_fit_T_max']  = 200 #linear fit to values obtained by allQ values fit
            dic['lin_T_max_for_allQ'] = dic['doster_lin_fit_T_max'] #until which T all Q are used, then the conditions of Q values are used
            if dic['doster_lin_fit_T_max'] > dic['lin_T_max_for_allQ']: 
                print ('Warning: doster_lin_fit_T_max(%i) > lin_T_max_for_allQ(%i)' % (dic['doster_lin_fit_T_max'],dic['doster_lin_fit_T_max']) )
            dic['doster_t_start_fix_d']  = 225 # only works if  doster_single_fit=True
            
            dic['doster_MSDG_starting_value_eq_lin_allQ'] = True
        
        #doster options END
        ##################
        
        #global options
        #print_output=True
        options_dic['fix_to_offset'] = config_dic['Global']['fix_to_offset']
        options_dic['no_weighting_in_fit'] = config_dic['Global']['no_weighting_in_fit']
        if (self._used_data_err==0).all():
            options_dic['no_weighting_in_fit'] = True
            if config_dic['Global']['no_weighting_in_fit'] != options_dic['no_weighting_in_fit']:
                warnings_list.append('Value [\'Global\'][\'no_weighting_in_fit\'] = False, but had to be changed to True because the error data does have 0 entries.')
            options_dic['no_data_error'] = True
            warnings_list.append('[\'no_data_error\']=True')
            
        options_dic['print_report']=config_dic['Global']['print_report'] #False
        #plot settings
        options_dic['plots_fitsAllT_Qrange_min-max']=config_dic['Global']['plots_fitsAllT_Qrange_min-max'] #[0,4.8] #min, max value of Qrange
        
        self.fitting_options_dic=options_dic
        if len(warnings_list) > 0:
            self._warnings_fitting_options=warnings_list
        
        return

    def _record_saving(self,results_lmfit_dic={}):
        self._record_nb=self._record_nb+1
        self._record_config_dic.append(copy.deepcopy(self.config_dic))
        self._record_fitting_options_dic.append(copy.deepcopy(self.fitting_options_dic))
        self._record_warnings_fitting_options.append(copy.deepcopy(self._warnings_fitting_options))
        self._record_warnings_fitting_process.append(copy.deepcopy(self._warnings_fitting_process))
        self._record_results_lmfit_dic.append(results_lmfit_dic)
        # !! _create_nice_results_dic uses created self._record_results_lmfit_dic !!
        self._record_nice_results_dic.append(self._create_nice_results_dic(record_nb=self._record_nb-1,silent=True))

    # def _load_saved_record(self):
    #     self.read_config_file()
    #     warnings_options,warnings_process=self._load_warnings_of_record_in_nice_results_dic()
    #     lmfit_dic=self.load_lmfit_results_local()

    #     self._record_nb=self._record_nb+1
    #     self._record_config_dic.append(copy.deepcopy(self.config_dic))
    #     self._record_fitting_options_dic.append(None)
    #     self._record_warnings_fitting_options.append(warnings_options) #could also be read from config_file if wished
    #     self._record_warnings_fitting_process.append(warnings_process) #could also be read from warnings file if wished
    #     self._record_results_lmfit_dic.append(lmfit_dic)
    #     # finally:
    #     #     if len(self._record_results_lmfit_dic) != self._record_nb:
    #     #         print('There have been problems ..., reverting changes')
    #     #         self._record_nb = self._record_nb-1
    #     #         for l in [self._record_config_dic,self._record_warnings_fitting_options,self._record_warnings_fitting_process,self._record_warnings_fitting_process,self._record_results_lmfit_dic]:
    #     #             try:
    #     #                 del l[self._record_nb]
    #     #             except IndexError:
    #     #                 pass


    def _load_warnings_of_record_in_nice_results_dic(self) -> list: #could also be done with file not with dictionary?!
        warnings_process=[]
        warnings_options=[]
        try:
            dic=self.load_nice_results_dic_local()
        except FileNotFoundError as err:
            print('Error : %s'%err.strerror)
            print('Warnings of fitting could not be loaded!')
            warnings_process.append('Possible warnings during fitting process could not be loaded!')
            warnings_options.append('Possible warnings produced for creating the fitting options could not be loaded!')
        
        key_warning='Warnings_during_fitting'
        if key_warning in dic: 
            for key in sorted(dic[key_warning]):
                warning=dic[key_warning][key]
                warnings_process.append(warning)
        
        key_warning='Warnings_fitting_options'
        if key_warning in dic: 
            for key in sorted(dic[key_warning]):
                warning=dic[key_warning][key]
                warnings_options.append(warning)

        if len(warnings_process) == 0:
            warnings_process = None
        if len(warnings_options) == 0:
            warnings_options = None

        return warnings_options,warnings_process
        
    @staticmethod
    def _sort_basic_options(dic,key_order=None):
        '''check input dictionary and order the keys
        '''
        order=['Global','GA','PK','Yi','Do'] #all available options
        key_order_out=None
        if key_order is None:
            key_order_out=order
            assert(len(dic)==len(order)), 'Something is not working here..., check the order list'
        else:
            add_keys=[]
            for key in sorted(dic):
                if key not in key_order:
                    add_keys.append(key)
            key_order_out=key_order+add_keys
        #check key_order_out keys
        for key in dic:
            if key not in key_order_out:
                assert(0), 'Something is not working here..., check the order list'
        for key in key_order_out:
            if key not in dic:
                assert(0), 'Something is not working here..., check the order list'
        return key_order_out
    
    def print_nb_of_records(self,):
        ''' Prints the number of records saved. '''
        print('There are %i records saved.'%self._record_nb)
        
    def print_config(self,record_nb=None):
        '''Prints config saved in config dictionary, either the current config or from a saved record.
            Parameters:
            record_nb : int, optional
                Define from which record number you want to read the config (-1 = last record).
                If None, current config is printed.
        '''
        dic=None
        warnings=None
        if record_nb is None:
            dic=self.config_dic
            warnings=self._warnings_fitting_options
        else:
            #check if record_nb is valid
            self._check_record_nb(record_nb,silent=False)
            dic=self._record_config_dic[record_nb]
            warnings=self._record_warnings_fitting_options[record_nb]
        
        key_order=self._sort_basic_options(dic)
        for section in key_order:
            for key in sorted(dic[section]):
                print ('[\'%s\'][\'%s\'] = %s'%(section,key,dic[section][key]))
        if warnings is not None:
            print('Warnings:\n %s'%warnings)
        return
    
    def get_config_dic(self,record_nb=None):
        '''Returns copy of config saved in config dictionary, either the current config or from a saved record.
            Parameters:
            record_nb : int, optional
                Define from which record number you want to read the config (-1 = last record).
                If None, current config is printed.
        '''
        dic=None
        warnings=None
        if record_nb is None:
            dic=self.config_dic
            warnings=self._warnings_fitting_options
        else:
            #check if record_nb is valid
            self._check_record_nb(record_nb,silent=False)
            dic=copy.deepcopy(self._record_config_dic[record_nb])
            warnings=self._record_warnings_fitting_options[record_nb]
    
        if warnings is not None:
            dic['Warnings']=warnings
            print('Warnings:\n %s'%warnings)

        return dic

    def run_fit(self,):
        ''' Fits the data set. 
            Fits are done with the config defined in self.config_dic dictionary. 
            self.config_dic can be set via read_config_file() or set_config_file().
            The results and configurations are saved in a new record. To get the number of available records use: print_nb_of_records.
        '''
        #initialize and set all variables needed
        self._define_fitting_regions()  #defines self.fitting_options_dic from config_dic
        options_dic=self.fitting_options_dic
        options_dic['info_added_during_fit']={}

        warnings_list=[]

        ndata =self._used_data.shape[0]
        used_q=self._used_q
        used_T=self._used_T
        
        used_q_squared=used_q**2
        
        used_data=self._used_data
        used_data_err=self._used_data_err
        used_data_log=self._used_data_log
        used_data_err_log=self._used_data_err_log

        dic_lmfit={}
        
        filename_save_str=str(self.name)
        print ('-'*40+'\n'+'Start fitting \"'+ str(self._raw_file_path)+'\":\n' + 'Save results as \"%s\"\n'% (filename_save_str))
        #################################################################################################
        ######################
        #linear = GA approx
        ######################
        
        #initialize linear fit paramters for GA
        dic_lmfit['GA'] = []
        GA_multi_params_list = []
        for i in range(ndata):
            multi_params = lmfit.Parameters()
            multi_params.add('intercept_%02d' % (i+1),value=-0.001, vary=True, max=options_dic['GA_intercept_max'])
            multi_params.add('slope_%02d' % (i+1),value=-0.001, vary=True)
            GA_multi_params_list.append(multi_params)
        
        #linear fit for Q-values defined by options_dic['qValues'] and options_dic['qValues_idx']
        #here NO offset=ln(intercept) is fixed!!
        print ('GA: Using leastsq algorithm:')
        for i in range(ndata):
            if self._raw_data_type=='sim':
                GA_multi_params_list[i]['intercept_%02d' % (i+1)].value=0.
                GA_multi_params_list[i]['intercept_%02d' % (i+1)].vary=False
                multi_fit = lmfit.minimize(lin_modelLmfitMultiBasic, GA_multi_params_list[i], args=(i,options_dic['models']['GA']['qValues_squared']), kws={'data':used_data_log[i,options_dic['models']['GA']['qValues_idx']],'eps':used_data_err_log[i,options_dic['models']['GA']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
            else:
                #first fit wih no constraints
                multi_fit = lmfit.minimize(lin_modelLmfitMultiBasic, GA_multi_params_list[i], args=(i,options_dic['models']['GA']['qValues_squared']), kws={'data':used_data_log[i,options_dic['models']['GA']['qValues_idx']],'eps':used_data_err_log[i,options_dic['models']['GA']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                #second fit with intercept fixed to zero IF one of the next conditions is fulfilled: 
                #1) if intercept is close to zero or even bigger: fix it to zero!!
                #2) if slope is larger than certain value, here 0.1 (corresponds to negative MSD): : fix it to zero!!
                if options_dic['GA_refit_if_interceptGreater0_OR_slopeGreater0']:
                    if multi_fit.params['intercept_%02d' % (i+1)] > -0.00001 or multi_fit.params['slope_%02d' % (i+1)] > 0.001:
                        GA_multi_params_list[i]['intercept_%02d' % (i+1)].value=0.
                        GA_multi_params_list[i]['intercept_%02d' % (i+1)].vary=False
                        multi_fit = lmfit.minimize(lin_modelLmfitMultiBasic, GA_multi_params_list[i], args=(i,options_dic['models']['GA']['qValues_squared']), kws={'data':used_data_log[i,options_dic['models']['GA']['qValues_idx']],'eps':used_data_err_log[i,options_dic['models']['GA']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
            dic_lmfit['GA'].append(multi_fit)
        
        #check if fit was fine:
        for i in range(ndata):
            if not dic_lmfit['GA'][i].success:
                warning_str='!! Warning: Linear fit (GA on log(data) vs Q^2) number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted!!'
                warnings_list.append(warning_str)
                print (warning_str)
            if not dic_lmfit['GA'][i].errorbars:
                warning_str='!! Warning: Linear fit (GA on log(data) vs Q^2) number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted, no errobar could be evaluated!!'
                warnings_list.append(warning_str)
                print (warning_str)
            for err in [dic_lmfit['GA'][i].params['slope_%02d' % (i+1)],dic_lmfit['GA'][i].params['intercept_%02d' % (i+1)],]:
                    if err.stderr is None:
                        err.stderr=0
                        warning_str='!! Warning: Linear fit (GA on log(data) vs Q^2) number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted, no error was evaluated for %s !!\n The value is set to 0.'%(err.name)
                        warnings_list.append(warning_str)
                        print (warning_str)
        
        #report fit result
        if options_dic['print_report']:
            print(lmfit.fit_report(dic_lmfit['GA'][-1]))

        #for easier plot and usage, calculate MSD and error with GA factor 3 defintion
        for i in range(ndata):
            dic_lmfit['GA'][i].params.add('MSD_F3_calc_%02d' % (i+1),value=dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].value*-3, vary=False)
            dic_lmfit['GA'][i].params['MSD_F3_calc_%02d' % (i+1)].stderr=dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].stderr*-3
        
        ######################
        # END - linear = GA approx - END
        ######################
        #################################################################################################
        
        
        #################################################################################################
        ######################
        # PK Model (PK_...)
        ######################
        options_dic['models']['PK']=options_dic['models']['PK']
        
        #initialize kn paramters for fitting
        PK_multi_params_list = []
        for i in range(ndata):
            multi_params = lmfit.Parameters()
            multi_params.add('offset_%02d' % (i+1),value=options_dic['offset_start_min_max'][0], vary=True, 
                            min=options_dic['offset_start_min_max'][1], 
                            max=options_dic['offset_start_min_max'][2])
            multi_params.add('sigma_%02d' % (i+1),value=options_dic['models']['PK']['PK_sigma_start-min-max'][0], vary=True, 
                            min=options_dic['models']['PK']['PK_sigma_start-min-max'][1], 
                            max=options_dic['models']['PK']['PK_sigma_start-min-max'][2])
            multi_params.add('beta_%02d' % (i+1),value=options_dic['models']['PK']['PK_beta_start-min-max'][0], vary=True, 
                            min=options_dic['models']['PK']['PK_beta_start-min-max'][1], 
                            max=options_dic['models']['PK']['PK_beta_start-min-max'][2])
            PK_multi_params_list.append(multi_params)
            
        #fitting with PK model
        #1) offset=ln(intercept) can be fixed by the first linear fit above (=GA) if options_dic['fix_to_offset'] = True
        #2) fit can be prefitted with DE (differential_evolution) algorithm (2nd if statement) if set in options_dic
        dic_lmfit['PK'] = []
        PK_multi_fit_DE_list = []
        for i in range(ndata):
                if options_dic['fix_to_offset'] == True:
                    PK_multi_params_list[i]['offset_%02d' % (i+1)].value=np.exp(dic_lmfit['GA'][i].params['intercept_%02d' % (i+1)].value)
                    PK_multi_params_list[i]['offset_%02d' % (i+1)].vary=False
                if 'DE_params' in options_dic:
                    if i ==0: print ('PK: Using DE algorithm:')
                    #initialize fitting with all parameters, but does not fit yet
                    multi_fit_DE_init = lmfit.Minimizer(pk_modelLmfitMultiBasic, PK_multi_params_list[i], fcn_args=(i,options_dic['models']['PK']['qValues']), fcn_kws={'data':used_data[i,options_dic['models']['PK']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['PK']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                    #fit with chosen method, here: method='differential_evolution'
                    multi_fit_DE=multi_fit_DE_init.scalar_minimize(method='differential_evolution',**options_dic['DE_params'])
                    #initialize fitting with all parameters obtained by DE, but does not fit yet
                    multi_fit_init = lmfit.Minimizer(pk_modelLmfitMultiBasic, multi_fit_DE.params, fcn_args=(i,options_dic['models']['PK']['qValues']), fcn_kws={'data':used_data[i,options_dic['models']['PK']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['PK']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                    #save DE fitting values
                    PK_multi_fit_DE_list.append(multi_fit_DE)
                else:
                    if i ==0: print ('PK: Using leastsq algorithm with initial parameters:')
                    multi_fit_init = lmfit.Minimizer(pk_modelLmfitMultiBasic, PK_multi_params_list[i], fcn_args=(i,options_dic['models']['PK']['qValues']), fcn_kws={'data':used_data[i,options_dic['models']['PK']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['PK']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                #fit multi_fit_init with chosen method, here: method='leastsq'  
                multi_fit=multi_fit_init.leastsq()
                ##multi_fit_init=lmfit.Minimizer(...) in the line above sets all parameters but does not fit, difference: in lmfit.Minimizer you have to put 'fcn_' before args and kws, 
                ##multi_fit_init.leastsq() = lmfit.minimize(method=leastsq,...see next line...) #method=leastsq is default, thus normally not stated in the fit 
                ##multi_fit = lmfit.minimize(pk_modelLmfitMultiBasic, PK_multi_params_list[i], args=(i,options_dic['models']['PK']['qValues']), kws={'data':used_data[i,options_dic['models']['PK']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['PK']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                dic_lmfit['PK'].append(multi_fit)
        
        #check if fit was fine:
        for i in range(ndata):
            if not dic_lmfit['PK'][i].success:
                warning_str='!! Warning: PK fit number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted!!'
                warnings_list.append(warning_str)
                print (warning_str)
            if not dic_lmfit['PK'][i].errorbars:
                warning_str='!! Warning: PK fit number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted, no errobar could be evaluated!!'
                warnings_list.append(warning_str)
                print (warning_str)
            for err in [dic_lmfit['PK'][i].params['beta_%02d' % (i+1)],dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)],]:
                    if err.stderr is None:
                        err.stderr=0
                        warning_str='!! Warning: PK fit number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted, no error was evaluated for %s !!\n The value is set to 0.'%(err.name)
                        warnings_list.append(warning_str)
                        print (warning_str)
        
        #report fit result of last entry(temperature)
        if options_dic['print_report']:
            print(lmfit.fit_report(dic_lmfit['PK'][-1]))
        
        #for easier plot and usage, calculate mspf and error with GA factor 3 defintion
        for i in range(ndata):
            sig=dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)].value
            sig_err=dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)].stderr
            dic_lmfit['PK'][i].params.add('MSD_F3_calc_%02d' % (i+1),value=sig**2*3, vary=False)
            dic_lmfit['PK'][i].params['MSD_F3_calc_%02d' % (i+1)].stderr=sig*sig_err*2*3
            #for standard deviaton of MSD
            beta=dic_lmfit['PK'][i].params['beta_%02d' % (i+1)].value
            beta_err=dic_lmfit['PK'][i].params['beta_%02d' % (i+1)].stderr
            msd3=dic_lmfit['PK'][i].params['MSD_F3_calc_%02d' % (i+1)].value
            dic_lmfit['PK'][i].params.add('STD_F3_calc_%02d' % (i+1),value=msd3/np.sqrt(beta), vary=False)
            dic_lmfit['PK'][i].params['STD_F3_calc_%02d' % (i+1)].stderr=3*np.sqrt( (2*sig*sig_err/np.sqrt(beta))**2 + (sig**2*0.5*beta_err/beta**(1.5))**2 )
        
        
        ######################
        # END PK Model (PK_...) END
        ######################
        ################################################################################################# 
            
        ################################################################################################# 
        ######################
        # Doster Model (doster_...)
        ######################
        if options_dic['use_fit_doster']:
            options_dic['models']['Do']=options_dic['models']['Do']
            options_dic['info_added_during_fit']['Do_model']={}

            ######################
            ####ONLY NEEDED IF DOSTER IS MODEL IS IN USE!:
            #linear fit for all Q-values to have a starting parameter for doster model msdGA
            #here offset=ln(intercept) can be fixed by the first linear fit above if options_dic['fix_to_offset'] = True
            #a max temperature can be given until which all Q are fitted, all temperatures above are fitted like done above (without correction at positive slopes or intercept>0)
            #if allQ is used the parameter all_Q is set to 1 if GA_x setting is used all_Q=0
            dic_lmfit['linAllQ'] = []
            for i in range(ndata):
                if options_dic['fix_to_offset'] == True:
                    GA_multi_params_list[i]['intercept_%02d' % (i+1)].value=dic_lmfit['GA'][i].params['intercept_%02d' % (i+1)].value
                    GA_multi_params_list[i]['intercept_%02d' % (i+1)].vary=False
                #if T < options_dic['lin_T_max_for_allQ'], fit all Q | if not fit like above on GA_x range
                if used_T[i] <= options_dic['models']['Do']['lin_T_max_for_allQ'] :
                    multi_fit = lmfit.minimize(lin_modelLmfitMultiBasic, GA_multi_params_list[i], args=(i,used_q_squared), kws={'data':used_data_log[i,:],'eps':used_data_err_log[i,:],'no_weighting':options_dic['no_weighting_in_fit']})
                    multi_fit.params.add('allQ_%02d' % (i+1),value=1, vary=False)
                    multi_fit.params['allQ_%02d' % (i+1)].stderr=0.
                else:
                    multi_fit = lmfit.minimize(lin_modelLmfitMultiBasic, GA_multi_params_list[i], args=(i,options_dic['models']['GA']['qValues_squared']), kws={'data':used_data_log[i,options_dic['models']['GA']['qValues_idx']],'eps':used_data_err_log[i,options_dic['models']['GA']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                    multi_fit.params.add('allQ_%02d' % (i+1),value=0, vary=False)
                    multi_fit.params['allQ_%02d' % (i+1)].stderr=0.
                dic_lmfit['linAllQ'].append(multi_fit)
            #for easier plot and usage, calculate MSD and error with GA factor 3 defintion
            for i in range(ndata):
                dic_lmfit['linAllQ'][i].params.add('MSD_F3_calc_%02d' % (i+1),value=dic_lmfit['linAllQ'][i].params['slope_%02d' % (i+1)].value*-3, vary=False)
                dic_lmfit['linAllQ'][i].params['MSD_F3_calc_%02d' % (i+1)].stderr=dic_lmfit['linAllQ'][i].params['slope_%02d' % (i+1)].stderr*-3
            ####ONLY NEEDED IF DOSTER IS MODEL IS IN USE!:
            ######################
        
            maxT=int(0)
            #maxT_value=used_T[0]
            msdGA = []
            msdGA_err = []
            for i in range(ndata):
                if used_T[i] <= options_dic['models']['Do']['doster_lin_fit_T_max'] :
                    maxT=i
                    #maxT_value=used_T[i]
                    msdGA.append(dic_lmfit['linAllQ'][i].params['slope_%02d' % (i+1)].value*-1)
                    msdGA_err.append(dic_lmfit['linAllQ'][i].params['slope_%02d' % (i+1)].stderr)
            msdGA=np.asarray(msdGA)
            msdGA_err=np.asarray(msdGA_err)
            msdGA_T=used_T[0:maxT+1]

            #print used_T[maxT],maxT_value, msdGA_T, msdGA, msdGA_err
                    
            doster_GA_params = lmfit.Parameters()
            doster_GA_params.add('intercept_%02d'%(int(1)),value=-0.02, vary=True)
            doster_GA_params.add('slope_%02d'%(int(1)),value=0.0001, vary=True)
            #get max T
            
            #if there are no T values under doster_lin_fit_T_max, msdGA is empty, 
            #--> if there is: fit linear fit to msdGA to get starting parameters for msdG in doster model
            if len(msdGA) > 0:
                doster_GA_fit = lmfit.minimize(lin_modelLmfitMultiBasic, doster_GA_params, args=(0,msdGA_T), kws={'data':msdGA})
                if options_dic['print_report']:
                    print(lmfit.fit_report(doster_GA_fit))
                options_dic['info_added_during_fit']['Do_model']['msdGA']=msdGA
                options_dic['info_added_during_fit']['Do_model']['msdGA_err']=msdGA_err
                options_dic['info_added_during_fit']['Do_model']['msdGA_T']=msdGA_T
                options_dic['info_added_during_fit']['Do_model']['msdGA_lmfit']=doster_GA_fit
            #initialize doster_ paramters for fitting
            doster_multi_params_list = []
            multi_params = lmfit.Parameters()
            for i in range(ndata):
                if options_dic['models']['Do']['doster_single_fit'] == False:
                    multi_params = lmfit.Parameters()
                multi_params.add('offset_%02d' % (i+1),value=options_dic['offset_start_min_max'][0], vary=True, 
                                min=options_dic['offset_start_min_max'][1], 
                                max=options_dic['offset_start_min_max'][2])
                #if used_T[i] <= 150. :
                #    multi_params.add('msdG_%02d' % (i+1),value=dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].value, vary=True, min=0.00001, max=10.)
                #else
                #    multi_params.add('msdG_%02d' % (i+1),value=1., vary=True, min=0.00001, max=10.)
                
                #if there are no T values under doster_lin_fit_T_max, msdGA is empty and no fit was done
                if (len(msdGA) > 0):  
                    multi_params.add('msdG_%02d' % (i+1),value=lin_modelLmfitMultiBasic(doster_GA_fit.params,0,used_T[i]), min=1e-6, max=2., vary=False) 
                    if options_dic['models']['Do']['doster_MSDG_starting_value_eq_lin_allQ'] == True:
                        multi_params['msdG_%02d' % (i+1)].vary = True
                        if multi_params['msdG_%02d' % (i+1)].value < 2e-6:
                            multi_params['msdG_%02d' % (i+1)].value=0.01
                else:
                    multi_params.add('msdG_%02d' % (i+1),value=0.1, min=0.000001, max=2., vary=True)
                #multi_params.add('msdG_%02d' % (i+1),value=dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].value*(-1), vary=False)
                #multi_params.add('msdG_%02d' % (i+1),value=0.1, vary=True)
                multi_params.add('p12_%02d' % (i+1),value=0.04, vary=True, min=0.0001, max=0.25) #changed from 0.1 to 0.04
                multi_params.add('d_%02d' % (i+1),value=1.5, vary=True, min=0.4, max=5.)
                if options_dic['models']['Do']['doster_single_fit'] == False:
                    doster_multi_params_list.append(multi_params)
                else:
                    if used_T[i] > options_dic['models']['Do']['doster_t_start_fix_d']: multi_params['d_%02d' % (i+1)].expr='d_%02d' % (i)
                
            if options_dic['models']['Do']['doster_single_fit'] == True:
                doster_multi_params_list.append(multi_params)
            
            dic_lmfit['Do'] = []
            doster_multi_fit_DE_list = []
            for i in range(ndata):
                #if options_dic['doster_single_fit'] = True we have only on item in list -> set index 'il' of list to 0
                il = i if (options_dic['models']['Do']['doster_single_fit'] == False) else 0
                if options_dic['fix_to_offset'] == True:
                    doster_multi_params_list[il]['offset_%02d' % (i+1)].value=np.exp(dic_lmfit['GA'][i].params['intercept_%02d' % (i+1)].value)
                    doster_multi_params_list[il]['offset_%02d' % (i+1)].vary=False
                if options_dic['models']['Do']['doster_single_fit'] == False:
                    if options_dic['models']['Do']['doster_d_fixed']:
                        doster_multi_params_list[i]['d_%02d' % (i+1)].value=options_dic['models']['Do']['doster_d_val']
                        doster_multi_params_list[i]['d_%02d' % (i+1)].vary=False
                    
                    #multi_fit = lmfit.minimize(doster_modelLmfitMultiBasic, doster_multi_params_list[i], args=(i,options_dic['qValues']), kws={'data':used_data[i,options_dic['qValues_idx']],'eps':used_data_err[i,options_dic['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                    if 'DE_params' in options_dic:
                        if i ==0: print ('Do: Using DE algorithm:')
                        multi_fit_DE_init = lmfit.Minimizer(doster_modelLmfitMultiBasic, doster_multi_params_list[i], fcn_args=(i,options_dic['models']['Do']['qValues']), fcn_kws={'data':used_data[i,options_dic['models']['Do']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['Do']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                        multi_fit_DE=multi_fit_DE_init.scalar_minimize(method='differential_evolution',**options_dic['DE_params'])
                        multi_fit_init = lmfit.Minimizer(doster_modelLmfitMultiBasic, multi_fit_DE.params, fcn_args=(i,options_dic['models']['Do']['qValues']), fcn_kws={'data':used_data[i,options_dic['models']['Do']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['Do']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                        doster_multi_fit_DE_list.append(multi_fit_DE)
                    else:
                        if i ==0: print ('Do: Using leastsq algorithm with initial parameters:')
                        multi_fit_init = lmfit.Minimizer(doster_modelLmfitMultiBasic, doster_multi_params_list[i], fcn_args=(i,options_dic['models']['Do']['qValues']), fcn_kws={'data':used_data[i,options_dic['models']['Do']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['Do']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                    multi_fit=multi_fit_init.leastsq()
                    
                    dic_lmfit['Do'].append(multi_fit)
            if options_dic['models']['Do']['doster_single_fit'] == True:
                multi_fit = lmfit.minimize(doster_modelLmfitMultiBasic_Multi, doster_multi_params_list[0], args=(options_dic['models']['Do']['qValues'],), kws={'data':used_data[:,options_dic['models']['Do']['qValues_idx']],'eps':used_data_err[:,options_dic['models']['Do']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                dic_lmfit['Do'].append(multi_fit)
            
            #check if fit was fine:
            for i in range(ndata):
                #if options_dic['models']['Do']['doster_single_fit'] = True we have only one item in list -> set index 'il' of list to 0
                il = i if (options_dic['models']['Do']['doster_single_fit'] == False) else 0
                if not dic_lmfit['Do'][il].success:
                    warning_str='!! Warning: Doster fit number i=%i (T=%i)'%(il,used_T[i]) + ' was not successfully fitted !!'
                    warnings_list.append(warning_str)
                    print (warning_str)
                if not dic_lmfit['Do'][il].errorbars:
                    warning_str='!! Warning: Doster fit number i=%i (T=%i)'%(il,used_T[i]) + ' was not successfully fitted, no errobar could be evaluated!!'
                    warnings_list.append(warning_str)
                    print (warning_str)
                for err in [dic_lmfit['Do'][il].params['msdG_%02d' % (i+1)],dic_lmfit['Do'][il].params['p12_%02d' % (i+1)],dic_lmfit['Do'][il].params['d_%02d' % (i+1)]]:
                    if err.stderr is None:
                        err.stderr=0
                        warning_str='!! Warning: Doster fit number i=%i (T=%i)'%(il,used_T[i]) + ' was not successfully fitted, no error was evaluated for %s !!\n The value is set to 0.'%(err.name)
                        warnings_list.append(warning_str)
                        print (warning_str)
                    
            
            #calc p1 and p2 with p12 and msd_tot, msd_tot is calculated without factor 3 (only one room direction) 
            for i in range(ndata):
                #if options_dic['models']['Do']['doster_single_fit'] = True we have only one item in list -> set index 'il' of list to 0
                il = i if (options_dic['models']['Do']['doster_single_fit'] == False) else 0
                
                # pq formula => used_q=-p/2. +- sqrt((p/2.)**2-q) for used_q**2 + p*used_q + q = 0 
                #here: p1+p2=1 and p12=p1*p2: p1*(1-p1)=p12 -> p1**2 - p1 + p12 = 0
                #      q = p12 and p = -1
                #      p1=1/2. +- ((-1/2.)**2 + p12)**0.5
                p12=dic_lmfit['Do'][il].params['p12_%02d' % (i+1)].value
                p1=0.5+(0.25-p12)**0.5
                p2=1.-p1 #or p2=0.5-(0.25-p12)**0.5
                dic_lmfit['Do'][il].params.add('p1_calc_%02d' % (i+1),value=p1, vary=False)
                dic_lmfit['Do'][il].params.add('p2_calc_%02d' % (i+1),value=p2, vary=False)
                
                #add error
                dic_lmfit['Do'][il].params['p1_calc_%02d' % (i+1)].stderr=0.5*(0.25-p12)**(-0.5)*dic_lmfit['Do'][il].params['p12_%02d' % (i+1)].stderr
                dic_lmfit['Do'][il].params['p2_calc_%02d' % (i+1)].stderr=dic_lmfit['Do'][il].params['p1_calc_%02d' % (i+1)].stderr
                
                #msd tot
                msd_tot=dic_lmfit['Do'][il].params['msdG_%02d' % (i+1)].value + (dic_lmfit['Do'][il].params['p12_%02d' % (i+1)].value*(dic_lmfit['Do'][il].params['d_%02d' % (i+1)].value)**2/3.)
                dic_lmfit['Do'][il].params.add('msd_calc_%02d' % (i+1),value=msd_tot, vary=False)
                #err = sqrt(a**2 + b**2 + c**2)
                temp_a_Gerr=dic_lmfit['Do'][il].params['msdG_%02d' % (i+1)].stderr
                temp_b_p12err=dic_lmfit['Do'][il].params['p12_%02d' % (i+1)].stderr * (dic_lmfit['Do'][il].params['d_%02d' % (i+1)].value)**2/3.
                temp_c_derr=dic_lmfit['Do'][il].params['d_%02d' % (i+1)].stderr * 2. * dic_lmfit['Do'][il].params['p12_%02d' % (i+1)].value * dic_lmfit['Do'][il].params['d_%02d' % (i+1)].value / 3.
                temp_value=temp_a_Gerr**2 + temp_b_p12err**2 + temp_c_derr**2
                dic_lmfit['Do'][il].params['msd_calc_%02d' % (i+1)].stderr=np.sqrt(temp_value)
        
                
            if options_dic['print_report']:
                print(lmfit.fit_report(dic_lmfit['Do'][-1]))

            #for easier plot and usage, calculate MSD and error with GA factor 3 defintion
            for i in range(ndata):
                dic_lmfit['Do'][i].params.add('MSD_F3_calc_%02d' % (i+1),value=dic_lmfit['Do'][i].params['msd_calc_%02d' % (i+1)].value*3, vary=False)
                dic_lmfit['Do'][i].params['MSD_F3_calc_%02d' % (i+1)].stderr=dic_lmfit['Do'][i].params['msd_calc_%02d' % (i+1)].stderr*3
        
        
        ######################
        # END - Doster Model (doster_...) - END
        ######################
        #################################################################################################
        
        #################################################################################################
        ######################
        # Yi Model (Yi_...) 
        ######################
        options_dic['models']['Yi']=options_dic['models']['Yi']
        
        #initialize Yi_ paramters for fitting
        Yi_multi_params_list = []
        for i in range(ndata):
            multi_params = lmfit.Parameters()
            multi_params.add('offset_%02d' % (i+1),value=options_dic['offset_start_min_max'][0], vary=True, 
                            min=options_dic['offset_start_min_max'][1], 
                            max=options_dic['offset_start_min_max'][2])
            multi_params.add('msd_%02d' % (i+1),value=options_dic['models']['Yi']['Yi_msd_start-min-max'][0], vary=True, 
                            min=options_dic['models']['Yi']['Yi_msd_start-min-max'][1], 
                            max=options_dic['models']['Yi']['Yi_msd_start-min-max'][2])
            multi_params.add('sigma_%02d' % (i+1),value=options_dic['models']['Yi']['Yi_sigma_start-min-max'][0], vary=True, 
                            min=options_dic['models']['Yi']['Yi_sigma_start-min-max'][1], 
                            max=options_dic['models']['Yi']['Yi_sigma_start-min-max'][2])
            
            Yi_multi_params_list.append(multi_params)
        
        dic_lmfit['Yi'] = []
        Yi_multi_fit_DE_list = []
        for i in range(ndata):
                if options_dic['fix_to_offset'] == True:
                    Yi_multi_params_list[i]['offset_%02d' % (i+1)].value=np.exp(dic_lmfit['GA'][i].params['intercept_%02d' % (i+1)].value)
                    Yi_multi_params_list[i]['offset_%02d' % (i+1)].vary=False
                
                if 'DE_params' in options_dic:
                    if i ==0: print ('Yi: Using DE algorithm:')
                    multi_fit_DE_init = lmfit.Minimizer(yi_modelLmfitMultiBasic, Yi_multi_params_list[i], fcn_args=(i,options_dic['models']['Yi']['qValues']), fcn_kws={'data':used_data[i,options_dic['models']['Yi']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['Yi']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                    multi_fit_DE=multi_fit_DE_init.scalar_minimize(method='differential_evolution',**options_dic['DE_params'])
                    ##alternative writing to the first two lines:
                    ##multi_fit_DE = lmfit.minimize(yi_modelLmfitMultiBasic, Yi_multi_params_list[i], method='differential_evolution', args=(i,options_dic['models']['Yi']['qValues']), kws={'data':used_data[i,options_dic['models']['Yi']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['Yi']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']}, **DE_params)
                    
                    ## code in case the fitting values obtained in DE mode are stuck at the minimum allowed value 
                    ##if multi_fit_DE.params['sigma_%02d' % (i+1)].value <= multi_fit_DE.params['sigma_%02d' % (i+1)].min+1e-03:
                    ##    multi_fit_DE.params['sigma_%02d' % (i+1)].value=2e-03
                    ##if multi_fit_DE.params['msd_%02d' % (i+1)].value <= multi_fit_DE.params['msd_%02d' % (i+1)].min+1e-03:
                    ##    multi_fit_DE.params['msd_%02d' % (i+1)].value=2e-03
                    
                    multi_fit = lmfit.minimize(yi_modelLmfitMultiBasic, multi_fit_DE.params, args=(i,options_dic['models']['Yi']['qValues']), kws={'data':used_data[i,options_dic['models']['Yi']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['Yi']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                    Yi_multi_fit_DE_list.append(multi_fit_DE)
                else:
                    if i ==0: print ('Yi: Using leastsq algorithm with initial parameters:')
                    multi_fit = lmfit.minimize(yi_modelLmfitMultiBasic, Yi_multi_params_list[i], args=(i,options_dic['models']['Yi']['qValues']), kws={'data':used_data[i,options_dic['models']['Yi']['qValues_idx']],'eps':used_data_err[i,options_dic['models']['Yi']['qValues_idx']],'no_weighting':options_dic['no_weighting_in_fit']})
                dic_lmfit['Yi'].append(multi_fit)
                ##for alternativ writing with multi_fit_init=lmfit.Minimzer(...) and multi_fit_init.leastsq() see PK model (also for more details of code)
        
        
        #check if fit was fine:
        for i in range(ndata):
            if not dic_lmfit['Yi'][i].success:
                warning_str='!! Warning: Yi fit number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted!!'
                warnings_list.append(warning_str)
                print (warning_str)
            if not dic_lmfit['Yi'][i].errorbars:
                warning_str='!! Warning: Yi fit number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted, no errobar could be evaluated!!'
                warnings_list.append(warning_str)
                print (warning_str)
            for err in [dic_lmfit['Yi'][i].params['msd_%02d' % (i+1)],dic_lmfit['Yi'][i].params['sigma_%02d' % (i+1)],]:
                    if err.stderr is None:
                        err.stderr=0
                        warning_str='!! Warning: Yi fit number i=%i (T=%i)'%(i,used_T[i]) + ' was not successfully fitted, no error was evaluated for %s !!\n The value is set to 0.'%(err.name)
                        warnings_list.append(warning_str)
                        print (warning_str)
        
        #print report on screen
        if options_dic['print_report']:
            print(lmfit.fit_report(dic_lmfit['Yi'][-1]))

        #for easier plot and usage, calculate MSD and error with GA factor 3 defintion
        for i in range(ndata):
            dic_lmfit['Yi'][i].params.add('MSD_F3_calc_%02d' % (i+1),value=dic_lmfit['Yi'][i].params['msd_%02d' % (i+1)].value*0.5, vary=False)
            dic_lmfit['Yi'][i].params['MSD_F3_calc_%02d' % (i+1)].stderr=dic_lmfit['Yi'][i].params['msd_%02d' % (i+1)].stderr*0.5
            #for standard deviaton of MSD
            dic_lmfit['Yi'][i].params.add('STD_F3_calc_%02d' % (i+1),value=dic_lmfit['Yi'][i].params['sigma_%02d' % (i+1)].value*0.5, vary=False)
            dic_lmfit['Yi'][i].params['STD_F3_calc_%02d' % (i+1)].stderr=dic_lmfit['Yi'][i].params['sigma_%02d' % (i+1)].stderr*0.5
        
        ######################
        # END - Yi Model (Yi_...) - END
        ######################
        #################################################################################################

        if len(warnings_list) > 0:
            self._warnings_fitting_process=warnings_list
        
        self._record_saving(dic_lmfit)
        
        #print summary of fit
        print ('-----')
        if options_dic['fix_to_offset']:
            print ('Values were fixed to offset of GA')
        if options_dic['no_weighting_in_fit']:
            print ('Fit was NOT weighted by errors!')
            if options_dic['no_data_error']:
                print ('... because there was no error given for data!')
        else:
            print ('Fit was weighted by errors!')
        print ('-----\n\"'+ str(filename_save_str)+'\" finished.')
        
        return

    def plot_results(self,record_nb=-1,save=False,close_all=False,save_path=None,silent=False,outputfile_type='png',outputfile_dpi=200):
        '''Plots the results of the fitted data set.
            Parameters:
            -----------
            record_nb : int, optional
                Define from which record number you want to plot the results. (-1 = last record).
            save : bool, optional
                If True, saves the plots in the default save directory (can be changed with set_save_dir() ) 
                    or in path given in "save_path" parameter.
            close_all : bool, optional
                If True, closes all plotted figures after exection. Suggested if parameter "save" = True.
            save_path : string, optional
                Directory where pltted figures are saved. If None, the default save directory (can be changed with set_save_dir() ) is used.
            silent : bool, optional
                If True, no output is printed to the terminal.
            outputfile_type : string, optional
                Define the type of your saved output, e.g. '.png', '.jpg', '.pdf'
            outputfile_dpi : int, optional
                Define the dpi (dots per inch) of your saved output, e.g. 200, 300, 600
        '''
        #checks
        self._check_record_nb(record_nb,silent=silent)

        #load dictionary results
        dic_lmfit=self._record_results_lmfit_dic[record_nb]
        options_dic=self._record_fitting_options_dic[record_nb]

        used_q=self._used_q
        used_q_squared=used_q**2
        used_Qmin=self._used_Qmin
        used_Qmax=self._used_Qmax
        used_T=self._used_T
        used_data=self._used_data
        used_data_err=self._used_data_err
        used_data_log=self._used_data_log
        used_data_err_log=self._used_data_err_log
        
        #plots for linear fits = GA
        fig_GA_fit_allT_list = []
        for nb_fig in range(0,int(len(used_T)/16)+1,1):
            fig1 = plt.figure(figsize=(20,10))
            nbplots=0
            for i in range(nb_fig*16,nb_fig*16+16,1):
                if i >= len(used_T): break
                nbplots=nbplots+1
                ft = fig1.add_subplot(4,4,nbplots)
                ft.errorbar(used_q_squared, used_data_log[i,:],yerr=used_data_err_log[i,:], marker='o',label='T=%i' % used_T[i])
                ft.errorbar(options_dic['models']['GA']['qValues_squared'], used_data_log[i,options_dic['models']['GA']['qValues_idx']][0],yerr=used_data_err_log[i,options_dic['models']['GA']['qValues_idx']][0], marker='o',label='fit region')
                GA_y_fit_temp = lin_modelLmfitMultiBasic(dic_lmfit['GA'][i].params, i, used_q**2)
                ft.plot(used_q**2,GA_y_fit_temp,label='Gaus3 %.2f' % (dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].value*-3))
                ft.set_xlim(xmin=-0.1,xmax=(np.max(options_dic['models']['GA']['qValues_squared'])*2))
                ft.set_ylim(ymax=0.05,ymin=(np.min(used_data_log[i,options_dic['models']['GA']['qValues_idx']])*3))
                ft.legend(prop={'size':10},loc='best')
                ft.set_xlabel('Q$^{2}$[$\\AA^{-2}$]')
                ft.set_ylabel('ln(EISF)')
            plot_title_string='GA model Qmin-Qmax: %.2f-%.2f$\\AA^{-1}$' %\
                                (options_dic['models']['GA']['qValues'][0],options_dic['models']['GA']['qValues'][-1])
            fig1.suptitle(plot_title_string)
            #make sure that there is at least one plot in figure
            if nbplots != 0:
                fig1.tight_layout()
                fig1.subplots_adjust(top=0.95)
                fig_GA_fit_allT_list.append(fig1)
        
        #all fits plot
        fig_fits_allT_list = []
        x_long=np.linspace(options_dic['plots_fitsAllT_Qrange_min-max'][0],options_dic['plots_fitsAllT_Qrange_min-max'][1],100)
        for nb_fig in range(0,int(len(used_T)/16)+1,1):
            fig1 = plt.figure(figsize=(20,10))
            nbplots=0
            for i in range(nb_fig*16,nb_fig*16+16,1):
                if i >= len(used_T): break
                nbplots=nbplots+1
                GA_y_fit = lin_modelLmfitMultiBasic(dic_lmfit['GA'][i].params, i, x_long**2)
                PK_y_fit = pk_modelLmfitMultiBasic(dic_lmfit['PK'][i].params, i, x_long)
                if options_dic['use_fit_doster']: 
                    #if options_dic['models']['Do']['doster_single_fit'] == True we have only on item in list -> set index 'il' of list to 0
                    il = i if (options_dic['models']['Do']['doster_single_fit'] == False) else 0
                    doster_y_fit = doster_modelLmfitMultiBasic(dic_lmfit['Do'][il].params, i, x_long)
                Yi_y_fit = yi_modelLmfitMultiBasic(dic_lmfit['Yi'][i].params, i, x_long)
                
                ft = fig1.add_subplot(4,4,nbplots)
                errorbar_plot=False
                if used_data_err.all()==0:
                    ft .plot(used_q, used_data[i, :], marker='o',  linestyle='None',label='T=%i' % used_T[i],ms=3,zorder=1)
                else:
                    ft .errorbar(used_q, used_data[i, :],yerr=used_data_err[i, :], marker='o',  linestyle='None',label='T=%i' % used_T[i],ms=3,zorder=1)
                    errorbar_plot=True
                
                ft .plot(x_long, np.exp(GA_y_fit), '-', label='Gauss3=%.2f' % (dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].value*-3),zorder=0)
                ft .plot(x_long, PK_y_fit, '-', label='MSPF3=%.2f' % (dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)].value**2*3),zorder=0)
                #il
                if options_dic['use_fit_doster']: 
                    ft .plot(x_long, doster_y_fit, '-', label='MSDG3=%.2f,\nMSDt3=%.2f' % (dic_lmfit['Do'][il].params['msdG_%02d' % (i+1)].value*3,dic_lmfit['Do'][il].params['msd_calc_%02d' % (i+1)].value*3),zorder=0)
                ft .plot(x_long, Yi_y_fit, '-', label='Yi MSD3=%.2f' % (dic_lmfit['Yi'][i].params['msd_%02d' % (i+1)].value*0.5),zorder=0)
                
                handles,labels = ft .get_legend_handles_labels()
                if errorbar_plot:
                    handles = [handles[-1]]+handles[0:-1]
                    labels = [labels[-1]]+labels[0:-1]
                ft.legend(handles,labels,prop={'size':10},loc='best')
                ft.set_xlabel('Q[$\\AA^{-1}$]')
                ft.set_ylabel('EISF')
                #if y max is bigger than value then set it to a maximum value; ft.axis() prints following array: (xmin,xmax,ymin,ymax)
                if ft.axis()[3] > 1.3:
                    maxVal=np.amax(used_data[i, :]-used_data_err[i, :])
                    ft.set_ylim(top=max(maxVal+0.1,1.3))
                if ft.axis()[2] < 0.3:
                    minVal=np.amin(used_data[i, :]-used_data_err[i, :])
                    ft.set_ylim(bottom=max(minVal-0.1,0))
            plot_title_string='FileIn: %s, Qmin: %i=%.2f$\\AA^{-1}$ Qmax: %i=%f$\\AA^{-1}$, GA Qmin-Qmax: %.2f-%.2f$\\AA^{-1}$, Yi Qmax: %.2f$\\AA^{-1}$, PK Qmax: %.2f$\\AA^{-1}$' %\
                                (self._raw_file_path,used_Qmin,used_q[0],used_Qmax,used_q[-1],options_dic['models']['GA']['qValues'][0],options_dic['models']['GA']['qValues'][-1]
                                ,options_dic['models']['Yi']['qValues'][-1],options_dic['models']['PK']['qValues'][-1]
                                )
            if options_dic['use_fit_doster']: 
                plot_title_string+=', doster Qmax: %i=%.2f$\\AA^{-1}$' % (options_dic['models']['Do']['qValues'][0],options_dic['models']['Do']['qValues'][-1])
            fig1.suptitle(plot_title_string)
            #make sure that there is at least one plot in figure
            if nbplots != 0:
                fig1.tight_layout()
                fig1.subplots_adjust(top=0.95)
                fig_fits_allT_list.append(fig1)
        
        if options_dic['use_fit_doster']:
                #plot fit of doster MSDGA
                if 'msdGA' in options_dic['info_added_during_fit']['Do_model']:
                    figGA_for_doster = plt.figure(figsize=(20,10))
                    msdGA_T=options_dic['info_added_during_fit']['Do_model']['msdGA_T']
                    GA_y_fit_temp = lin_modelLmfitMultiBasic(options_dic['info_added_during_fit']['Do_model']['msdGA_lmfit'].params, 0, msdGA_T)
                    plt.plot(msdGA_T,GA_y_fit_temp)
                    plt.errorbar(msdGA_T,options_dic['info_added_during_fit']['Do_model']['msdGA'],yerr=options_dic['info_added_during_fit']['Do_model']['msdGA_err'])

        elinewidth=0.7*plt.rcParams['lines.linewidth']

        #MSD3 plot
        fig_MSD3=plt.figure(figsize=(10,10))
        yS_GA,yS_GAErr=[],[]
        yS_Yi,yS_YiErr=[],[]
        yS_PK,yS_PKErr=[],[]
        yS_doster,yS_dosterErr=[],[]
        for i in range(0,len(used_T),1):
            yS_GA.append(dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].value*-3)
            yS_GAErr.append(dic_lmfit['GA'][i].params['slope_%02d' % (i+1)].stderr*-3)
            yS_Yi.append(dic_lmfit['Yi'][i].params['msd_%02d' % (i+1)].value*0.5)
            yS_YiErr.append(dic_lmfit['Yi'][i].params['msd_%02d' % (i+1)].stderr*0.5)
            yS_PK.append(dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)].value**2*3)
            yS_PKErr.append(dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)].stderr*dic_lmfit['PK'][i].params['sigma_%02d' % (i+1)].value*2*3)
            if options_dic['use_fit_doster']: 
                #if options_dic['models']['Do']['doster_single_fit'] == True we have only on item in list -> set index 'il' of list to 0
                il = i if (options_dic['models']['Do']['doster_single_fit'] == False) else 0
                yS_doster.append(dic_lmfit['Do'][il].params['msdG_%02d' % (i+1)].value*3)
                yS_dosterErr.append(dic_lmfit['Do'][il].params['msdG_%02d' % (i+1)].stderr*3)
        plt.errorbar(used_T,yS_GA,yerr=yS_GAErr,elinewidth=elinewidth,marker='o')
        plt.errorbar(used_T,yS_PK,yerr=yS_PKErr,elinewidth=elinewidth,marker='o')
        plt.errorbar(used_T,yS_Yi,yerr=yS_YiErr,elinewidth=elinewidth,marker='o')
        if options_dic['use_fit_doster']: plt.errorbar(used_T,yS_doster,yerr=yS_dosterErr,elinewidth=elinewidth,marker='o')
        plt.legend(('GA MSD3','PK MSD3','Yi MSD3','Do MSD3'), loc=2)
        plt.ylim(ymax=np.max(yS_PK+yS_Yi+yS_GA)+0.2,ymin=np.min([0,np.min(yS_PK+yS_Yi+yS_GA)-0.1]) )
        plt.xlabel('T [K]')
        plt.ylabel('MSD [$\\AA^{^2}$]')

        if not save: #only plotted if not saved
            #plot EISF(Q=0) of GA model
            fig_EISF=plt.figure(figsize=(10,10))
            y,y_Err=[],[]
            for i in range(0,len(used_T),1):
                y.append(np.exp(dic_lmfit['GA'][i].params['intercept_%02d' % (i+1)].value))
                y_Err.append(np.exp(dic_lmfit['GA'][i].params['intercept_%02d' % (i+1)].stderr))
            #plt.plot(used_T,y)
            plt.errorbar(used_T,y,y_Err,elinewidth=elinewidth,marker='o')
            plt.xlabel('T [K]')
            plt.ylabel('EISF(Q=0)')
            plt.ylim(ymax=np.max(y)+0.2,ymin=0)

            if options_dic['use_fit_doster']:
                #figure for allQ fits
                plt.figure()
                for i in range(np.maximum(0,len(used_T)-8),len(used_T),1):
                    plt.errorbar(used_q_squared, used_data_log[i,:],yerr=used_data_err_log[i,:], marker='o', label='T=%i' % (used_T[i]),elinewidth=elinewidth)
                plt.legend()
            ###################################################

            #STD3 plot
            fig_STD3=plt.figure(figsize=(10,10))
            yS_Yi,yS_YiErr=[],[]
            yS_PK,yS_PKErr=[],[]
            for i in range(0,len(used_T),1):
                yS_PK.append(dic_lmfit['PK'][i].params['STD_F3_calc_%02d'%(i+1)].value)
                yS_PKErr.append(dic_lmfit['PK'][i].params['STD_F3_calc_%02d'%(i+1)].stderr)
                yS_Yi.append(dic_lmfit['Yi'][i].params['STD_F3_calc_%02d'%(i+1)].value)
                yS_YiErr.append(dic_lmfit['Yi'][i].params['STD_F3_calc_%02d'%(i+1)].stderr)
            plt.errorbar(used_T,yS_PK,yerr=yS_PKErr,label='PK STD3',elinewidth=elinewidth,marker='o')
            plt.errorbar(used_T,yS_Yi,yerr=yS_YiErr,label='Yi STD3',elinewidth=elinewidth,marker='o')
            plt.legend(loc=2)
            plt.ylim(ymax=np.max(yS_PK+yS_Yi)+0.2,ymin=0)
            plt.xlabel('T [K]')
            plt.ylabel('STD [$\\AA^{^2}$]')
            
        #save plots if wished
        if save:
            #initialize and set all variables needed
            if save_path is None:
                save_location=self._save_dir_path
            else:
                save_location=self._prepare_save_dir(save_path,no_sub_folder=True)
            str_fixed='' if (options_dic['fix_to_offset'] == False) else 'fixed-'
            filename_save_str=str(self.name)
            for i,fig1 in enumerate(fig_fits_allT_list):
                fig1.savefig(save_location / ('fig_fits_allT-%02d-' % (i) +str_fixed+filename_save_str+'.%s'%outputfile_type), dpi=outputfile_dpi)
            for i,fig1 in enumerate(fig_GA_fit_allT_list):
                fig1.savefig(save_location / ('fig_GA_fit_allT-%02d-' % (i) +str_fixed+filename_save_str+'.%s'%outputfile_type), dpi=outputfile_dpi)
            if options_dic['use_fit_doster']: 
                figGA_for_doster.savefig(save_location / ('testDosterLinFit-'+str_fixed+filename_save_str+'.%s'%outputfile_type), dpi=outputfile_dpi)
            fig_MSD3.savefig(save_location / ('fig_MSD3'+str_fixed+filename_save_str+'.%s'%outputfile_type), dpi=outputfile_dpi)

            if not silent:
                print('Plots saved in folder \"%s\" with suffix \"%s\".'%(save_location,filename_save_str))
            
        if close_all:
            plt.close('all')
        
        return

    def save_results(self,record_nb=-1,save_path=None,silent=False):
        '''Saves the results of the fitted data set to two text files (prefix+'.txt' and prefix+'-vals.txt'). prefix=name_data_set + model_type
            Parameters:
            -----------
            record_nb : int, optional
                Define from which record number you want to plot the results. (-1 = last record).
            save_path : string, optional
                Directory where text files are saved. If None, the default save directory (can be changed with set_save_dir() ) is used.
            silent : bool, optional
                If True, no output is printed to the terminal.
        '''
        #checks
        self._check_record_nb(record_nb,silent=silent)
        #load dictionary results
        dic_lmfit=self._record_results_lmfit_dic[record_nb]
        options_dic=self._record_fitting_options_dic[record_nb]
        #initialize and set all variables needed
        if save_path is None:
            save_location=self._save_dir_path
        else:
            save_location=self._prepare_save_dir(save_path,no_sub_folder=True)
        str_fixed='' if (options_dic['fix_to_offset'] == False) else 'fixed-'
        filename_save_str=str(self.name)
        used_T=self._used_T

        #write warnings if they exist, if not, delete existing warning file
        warnings=self._record_warnings_fitting_process[record_nb]
        log_file=save_location / ('Warnings_fitting_process-'+str_fixed+filename_save_str+'.log')
        write_or_delete_warning(log_file,warnings=warnings)
        if warnings is not None:
            print('Warning:There have been warnings during the fitting process! Check created log file! (%s)'%log_file)
            if not silent:
                for w in warnings:
                    print(w)

        models_list=['GA','PK','Yi']
        for model in models_list:
            output_name=str(save_location / (model+'-'+str_fixed+filename_save_str))
            write_fit_report (output_name + '.txt'     ,dic_lmfit[model],used_T, options_dic['models'][model]['qValues'])
            write_fit_report2(output_name + '-vals.txt',dic_lmfit[model],used_T, options_dic['models'][model]['qValues'])
    
        #write_fit_report(save_location / ('PK-'+str_fixed+filename_save_str+'.txt'),dic_lmfit['PK'],used_T)
        #write_fit_report2(save_location / ('PK-'+str_fixed+filename_save_str+'-vals.txt'),dic_lmfit['PK'],used_T)
        
        #write_fit_report(save_location / ('Yi-'+str_fixed+filename_save_str+'.txt'),dic_lmfit['Yi'],used_T)
        #write_fit_report2(save_location / ('Yi-'+str_fixed+filename_save_str+'-vals.txt'),dic_lmfit['Yi'],used_T)
        
        if options_dic['use_fit_doster']:
            write_fit_report(save_location / ('doster-'+str_fixed+filename_save_str+'.txt'),dic_lmfit['Do'],used_T, options_dic['models']['Do']['qValues'])
            write_fit_report3(save_location / ('doster-'+str_fixed+filename_save_str+'-vals.txt'),dic_lmfit['Do'],used_T) if (options_dic['models']['Do']['doster_single_fit'] == True) else write_fit_report2(save_location / ('doster-'+str_fixed+filename_save_str+'-vals.txt'),dic_lmfit['Do'],used_T, options_dic['models']['Do']['qValues'])
            #for linAllQ
            write_fit_report(save_location / ('linAllQ-'+str_fixed+filename_save_str+'.txt'),dic_lmfit['linAllQ'],used_T)
            write_fit_report2(save_location / ('linAllQ-'+str_fixed+filename_save_str+'-vals.txt'),dic_lmfit['linAllQ'],used_T)
            #for fits for msdGA of doster
            if 'msdGA' in options_dic['info_added_during_fit']['Do_model']:
                msdGA_lmfit=options_dic['info_added_during_fit']['Do_model']['msdGA_lmfit']
                msdGA_T=options_dic['info_added_during_fit']['Do_model']['msdGA_T']
                write_fit_report(save_location / ('doster_msdGA_fit-'+filename_save_str+'.txt'),[msdGA_lmfit,],msdGA_T)


        if not silent:
            print('Data saved in folder \"%s\" with suffix \"%s\".'%(save_location,filename_save_str))
        return
            
    def save_input(self,save_path=None,silent=False):
        '''Saves the raw data and if used data is different, the dictionary of the used data
            Parameters:
            -----------
            save_path : string, optional
                Directory where text files are saved. If None, the default save directory (can be changed with set_save_dir() ) is used.
            silent : bool, optional
                If True, no output is printed to the terminal.
        '''
        #initialize and set all variables needed
        save_tmp=None
        if save_path is None:
            save_tmp=self._save_dir_path
        else:
            save_tmp=Path(save_path)
        save_location=self._prepare_save_dir(save_tmp / ('python_data_' + self.name),no_sub_folder=True)

        #save raw data
        save_data(save_location / 'raw_q',self._raw_q)
        save_data(save_location / 'raw_T',self._raw_T)
        save_data(save_location / 'raw_data',self._raw_data)
        save_data(save_location / 'raw_data_err',self._raw_data_err)

        if self._dic_transform_rawData_to_usedData is not None:
            save_data(save_location / 'dic_data_to_use',self._dic_transform_rawData_to_usedData)
        else:
            #remove file, since dic should not exist any more with this config!
            remove_data(save_location / 'dic_data_to_use')

        if not silent:
            print('Input data saved in folder \"%s%s\".'%(save_location,Path('/')))
        return
    
    def save_lmfit_results(self,record_nb=-1,save_path=None,silent=False):
        ''' Save dictionary of lmfit results of given record number as pickle file.

            Parameters:
            -----------
            record_nb : int, optional
                Define from which record number you want to plot the results. (-1 = last record).
            save_path : string, optional
                Directory where the pickle file is saved. If None, the default save directory (can be changed with set_save_dir() ) is used.
            silent : bool, optional
                If True, no output is printed to the terminal.
        '''
        #checks
        self._check_record_nb(record_nb,silent=silent)
        #load dictionary results
        dic_lmfit=self._record_results_lmfit_dic[record_nb]

        #initialize and set all variables needed
        if save_path is None:
            save_location=self._save_dir_path
        else:
            save_location=self._prepare_save_dir(save_path,no_sub_folder=True)
        file_save=save_location / ('lmfit_dic-%s.pickle'%self.name )

        if not silent:
            print('Saving lmfit dictionary in file %s.'%file_save)

        save_pickle(dic_lmfit,file_save)

        return

    @staticmethod
    def load_lmfit_results(loadfile) -> dict:
        ''' Loads dictionary of lmfit results (pickle file) and returns the dictionary file.

            Parameters:
            -----------
            loadfile : string
                Filename of lmfit results dictionary with or without supported suffix.
        '''
        file_load=Path(loadfile)
        file_type='pickle'
        if file_load.suffix == '':
            file_load=file_load.with_suffix('.%s'%file_type)
        else:
            if file_load.suffix != file_type:
                raise TypeError('File format type "%s" not supported. Use ".%s" file.'%(file_load.suffix,file_type))
        
        print('Loading lmfit results dic file "%s".'%file_load)

        return load_pickle(file_load)

    def load_lmfit_results_local(self) -> dict:
        ''' Loads dictionary of lmfit results (pickle file) saved in default save path and returns the dictionary file.
        '''
        file_load=self._save_dir_path / ('lmfit_dic-%s'%self.name )

        return self.load_lmfit_results(loadfile=file_load)

    def save_nice_results_dic(self,record_nb=-1,file_type='json',save_path=None,silent=False):
        ''' Save dictionary of nice results of given record number as .pickle or .json file.

            Parameters:
            -----------
            record_nb : int, optional
                Define from which record number you want to plot the results. (-1 = last record).
            file_type : 'pickle' or 'json', optional
                Define the file type of the saved dictionary file.
            save_path : string, optional
                Directory where the pickle file is saved. If None, the default save directory (can be changed with set_save_dir() ) is used.
            silent : bool, optional
                If True, no output is printed to the terminal.
        '''
        allowed_file_types=['pickle','json']
        if file_type not in allowed_file_types:
            raise ValueError('Input variable \"file_type\" is not in the list of allowed file types: %s'%allowed_file_types)
        dic_out=self.get_nice_results_dic(record_nb=record_nb,silent=silent)

        #initialize and set all variables needed
        if save_path is None:
            save_location=self._save_dir_path
        else:
            save_location=self._prepare_save_dir(save_path,no_sub_folder=True)
        file_save=(save_location / ('nice_results_dic-%s'%self.name )).with_suffix('.%s'%file_type)

        if not silent:
            print('Saving nice results dic in file %s.'%file_save)

        if file_type == 'pickle':
            save_pickle(dic_out,file_save)
        elif file_type == 'json':
            save_json(dic_out,file_save)
        
        return

    @staticmethod
    def load_nice_results_dic(loadfile) -> dict:
        ''' Loads dictionary of nice results (pickle or json file) and returns the dictionary file.

            Parameters:
            -----------
            loadfile : string
                Filename of nice results dictionary with or without supported suffix.
        '''
        allowed_file_types=['pickle','json']
        file_load=Path(loadfile)
        if file_load.suffix == '':
            file_type_load=None
            #check which suffix to load, pickle is default
            for file_type in allowed_file_types:
                if file_load.with_suffix('.%s'%file_type).is_file():
                    file_type_load=file_type
                    break #if one file is fould, break
            if file_type_load is None:
                raise FileNotFoundError('File:  \"%s\" + \".json\" or \".pickle\" not found.'%file_load)
            else:
                file_load=file_load.with_suffix('.%s'%file_type_load)
        else:
            if not file_load.suffix in allowed_file_types:
                raise TypeError('File format type "%s" not supported.'%file_load.suffix)
        print('Loading nice results dic file "%s".'%file_load)

        dic_out=None
        if file_load.suffix == '.pickle':
            dic_out=load_pickle(file_load)
        elif file_load.suffix == '.json':
            dic_out=load_json(file_load,convert_lists_to_numpy=True)
    
        return dic_out

    def load_nice_results_dic_local(self) -> dict:
        ''' Loads dictionary of nice results (pickle or json file) saved in default save path and returns the dictionary file.
        '''
        file_load=self._save_dir_path / ('nice_results_dic-%s'%self.name )

        return self.load_nice_results_dic(loadfile=file_load)
    
    def save_all(self,record_nb=-1,save_path=None,plot=True,silent=True):
        '''Saves data set, config, results and if wanted also figures.
            Parameters:
            -----------
            record_nb : int, optional
                Define from which record number you want to save the results. (-1 = last record).
            save_path : string, optional
                Base directory where results are saved. If None, the default save directory (can be changed with set_save_dir() ) is used.
            plot : bool, optional
                If False, figures ar not plotted and are not saved.
            silent : bool, optional
                If False, all output is printed to the terminal.
        '''
        self.save_results(record_nb=record_nb,save_path=save_path,silent=False)
        self.save_nice_results_dic(record_nb=record_nb,file_type='json',save_path=save_path,silent=silent)
        self.save_config_file(record_nb=record_nb,save_path=save_path,silent=silent)
        self.save_input(save_path=save_path,silent=silent)
        if plot:
            self.plot_results(record_nb=record_nb,save=True,close_all=True,save_path=save_path,silent=silent)

    def print_diff_in_config(self,record_nb1=0,record_nb2=-1,all=False,record_nb_ref=0):
        ''' Prints the difference between the config of two records.
            For differences between two different config dictionaries of different samples, use print_diff_between_two_dics()

            Parameters:
            -----------
            record_nb1 : int, optional
                Record number of first config to compare. ["0" = first config, "-1" = last config]
            record_nb2 : int, optional
                Record number of second config to compare. ["0" = first config, "-1" = last config]
            all : bool, optional
                Get differences of configs of all records. First record is the reference config.
            record_nb_ref : int, optional
                Record number of reference config -> all available configs are compared to this config. ["0" = first config, "-1" = last config]
        '''

        nb_records=self._record_nb
        if nb_records < 2:
            raise ValueError('There are not at least two records to compare.')

        if not all:
            self._check_record_nb(record_nb1,silent=True)
            self._check_record_nb(record_nb2,silent=True)

            d1=self._record_config_dic[record_nb1]
            d2=self._record_config_dic[record_nb2]
            
            print('Differences in config between record number "%i" (=d1) and "%i" (=d2):'%(record_nb1,record_nb2))
            self.print_diff_between_two_dics(d1,d2,as_string=False)
        else:
            nb_ref=record_nb_ref
            self._check_record_nb(nb_ref,silent=True)
            if nb_ref == -1:
                nb_ref=nb_records-1
            d_ref=self._record_config_dic[nb_ref]
            for idx,d2 in enumerate(self._record_config_dic):
                if idx != nb_ref:
                    print('Differences in config between record number "%i" (=d1) and "%i" (=d2):'%(nb_ref,idx))
                    self.print_diff_between_two_dics(d_ref,d2,as_string=False)
                

    @staticmethod
    def print_diff_between_two_dics(d1,d2,as_string=False):
        ''' Prints the difference between two dictionaries d1 and d2; d1 and d2 can be interchanged.
            Only works/tested with config_dic and fitting_dic

            Parameters:
            -----------
            d1 : dict, first dictionary
            d2 : dict, second dictionary
            as_string : bool, optional
                If True, function returns string, else the result is printed to stdout (normally terminal).
        '''
        from io import StringIO  # Python3
        old_stdout = sys.stdout #save old output
    
        #  result1 = StringIO() will store everything that is sent to the standard output
        try:
            result1 = StringIO()
            sys.stdout = result1
            print_diff_in_config_1D(d1, d2, path='Diff in')
            result2 = StringIO()
            sys.stdout = result2
            print_diff_in_config_1D(d2, d1, path='Diff in')
        #make sure that output is again on old output, even if there was an error!!
        finally:
            sys.stdout = old_stdout  # Redirect again the std output to old output

        # Then, get the stdout like a string and process it!
        result_string1 = result1.getvalue()
        result_string2 = result2.getvalue()
        #rename d1 to d2 and d2 to d1 for second dic, must be temporarily changed before!
        result_string2=result_string2.replace('d1','tmp8239_a2-adg-_1')
        result_string2=result_string2.replace('d2','tmp8239_a2-adg-_2')
        result_string2=result_string2.replace('tmp8239_a2-adg-_1','d2')
        result_string2=result_string2.replace('tmp8239_a2-adg-_2','d1',)
        
        result_output=None
        #if the dictrionaries are different, one dic should have more entries
        if len(result_string1)>=len(result_string2):
            #check if they are not different infos in the two results
            for s in result_string2.split('\n'):
                if s not in result_string1:
                    assert(0),s
            result_output=result_string1
        else:
            #check if they are not different infos in the two results
            for s in result_string1.split('\n'):
                if s not in result_string2:
                    assert(0),s
            result_output=result_string2
        
        if as_string:
            return result_output
        else:
            print(result_output)


    ## define readable and changable variables
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = str(value)

    ## define only readable variables:

    #If defined with deepcopy, then only _config_dic can write new properties which are saved, NOT config_dic!!
    @property
    def config_dic(self):
        return copy.deepcopy(self._config_dic)

    @property
    def raw_data_type(self):
        return copy.copy(self._raw_data_type)
    # @raw_data_type.setter
    # def raw_data_type(self, value):
    #     self._raw_data_type = value
    # @raw_data_type.deleter
    # def raw_data_type(self)
    #     del self._raw_data_type
    
    @property
    def raw_file_path(self):
        return copy.copy(self._raw_file_path)
    
    @property
    def raw_T(self):
        return copy.copy(self._raw_T)

    @property
    def raw_q(self):
        return copy.copy(self._raw_q)

    @property
    def raw_data(self):
        return copy.copy(self._raw_data)

    @property
    def raw_data_err(self):
        return copy.copy(self._raw_data_err)

    @property
    def used_T(self):
        return copy.copy(self._used_T)

    @property
    def used_q(self):
        return copy.copy(self._used_q)

    @property
    def used_data(self):
        return copy.copy(self._used_data)
    
    @property
    def used_data_err(self):
        return copy.copy(self._used_data_err)

    @property
    def used_data_log(self):
        return copy.copy(self._used_data_log)

    @property
    def used_data_err_log(self):
        return copy.copy(self._used_data_err_log)

    @property
    def used_Qmin(self):
        return copy.copy(self._used_Qmin)

    @property
    def used_Qmax(self):
        return copy.copy(self._used_Qmax)
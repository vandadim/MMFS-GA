"""
Created on Thu Jul  7 18:10:46 2022
@author: Vandad Imami

This script runs the MMFS-GA algorithm to optimize a Multimodal feature selection problem
using linear classifier. 
Linear Discriminant Analysis (LDA) for binary classification and Multinomial Logistic regression
for multicalss classification.
# This script loads each dataset from CSV files, where the path to the data files need to be specified by inputfiles variable.
"""
import random, os
import numpy as np
import pandas as pd
import mmfs2GA as mmfs
from multiprocessing import Event, Pipe, Process, Queue
from collections import deque
from numpy import genfromtxt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection 
from sklearn.metrics import balanced_accuracy_score

###############################################################################
"""                      Define function to load Data                       """    
###############################################################################
def load_data(file):
    """
    Loads data from a csv file into a pandas dataframe and returns a dictionary
    containing the dataframe values, class values, and column names.
    
    Args:
    - file (str): The file path of the csv file to load
    
    Returns:
    - data (dict): A dictionary containing the following keys:
        - 'dat': A tuple containing the dataframe values, class values, and column names
    """
    
    dat = pd.read_csv(file)
    dat.head()
    dat_class = dat['class']
    del dat['class']
    
    # Store dataframe values, class values, and column names in a tuple
    datac = (dat.iloc[:,:].values, dat_class[:].values, (dat.columns).values)
    return {'dat': datac}

###############################################################################
"""              Define function to evaluate solution                       """    
###############################################################################
def check_pop(solution, all_pop, dat,nf, nb):
    """
    Evaluates a solution using linear classifier.
    LDA for binary classification and Multinomial Logistic regression
    for multicalss classification.
    
    Args:
    - solution (np.ndarray): An array of binary values representing the selected features
    - all_pop (np.ndarray): An array containing all possible feature selected from different views
    - dat (list): A list of tuples, each containing the training data and class labels
    - nf (int): The number of folds for cross-validation
    - nb (int): The number of times to repeat the cross-validation process
    
    Returns:
    - value (int): accuracy of selected features
    - feature size (int): number of selected features
    - selected features(np.ndarray): selected features
    """
    
    solution=np.array(solution)
    all_data = np.concatenate([dt[0] for dt in dat],axis=1)
    label = dat[0][1]
    chromosome_mask=[]
    
    # Select the features based on the binary mask
    for i in range(len(solution)):
        chromosome_mask=np.append(chromosome_mask,all_pop[i][solution[i]],axis=0)
    con_ind = [i for (i, b) in enumerate(chromosome_mask) if b == 1]
    new_data = all_data[:,con_ind]
    mean_scores = []
    for param in range(nb):        
        scores = []        
        kfolds = model_selection.StratifiedKFold(n_splits=nf, 
                                                 shuffle=True, 
                                                 random_state=None)
        for train_index, test_index in kfolds.split(new_data,label):
            
            X_train, X_test = new_data[train_index],new_data[test_index]
            y_train, y_test = label[train_index], label[test_index]            
            cl_rf = LinearDiscriminantAnalysis()
            cl_rf.fit(X_train, y_train)
            y_pred=cl_rf.predict(X_test)
            scores.append(balanced_accuracy_score(y_test, y_pred))                
        mean_scores.append(np.mean(scores))            
    value = np.mean(mean_scores)
    
    return  value, sum(chromosome_mask),chromosome_mask
###############################################################################
###############################################################################
def final_result(result1_list,result3_list,dat):
    """
    Evaluates all solutions and returns the best solution and feature size based on accuracy.
    
    Args:
    - result1_list (list): A list of arrays representing the selected features for each solution
    - result3_list (list): A list of arrays containing all possible feature indices for each solution
    - dat (list): A list of tuples, each containing the training data and class labels
    
    Returns:
    - best_solution (tuple): A tuple representing the best solution, which is an array of binary values
    - fe_size (int): The number of selected features for the best solution
    """
    # Set initial values for best solution accuracy and feature size
    initial_Acc=0
    initial_Size=dat[0][0].size
    # Iterate through all solutions and evaluate them using correct classifier
    for i in range (len(result1_list)):
        solution = result1_list[i]
        all_pop = result3_list[i] 
        [accuracy,fe_size, selec_features]=check_pop(solution, all_pop, dat,nf=10, nb=10)
        if accuracy>initial_Acc or (accuracy == initial_Acc and fe_size < initial_Size):
            best_Solution=solution            
            initial_Acc=accuracy
            initial_Size=fe_size
            best_select_features=selec_features
    np.savetxt('SelectedFeatures.csv', best_select_features)
    return best_Solution,initial_Size
                             
###############################################################################
"""                      Define MMFS-GA Algorithm                           """    
###############################################################################     
def mmfsga(inputfile, outputdir, real_A=None, real_B=None, ngen=1000, npop=200, numN=6):
    """
    This script runs the 'ivfs' function from the 'mmfs2GA' module on multiple processes,
    using a set of input parameters and input files.
    Parameters:
        inputfile (list): List of input files containing data in CSV format
        outputdir (str): Path to the output directory where the MMFSGA results will be saved
        REAL_A (ndarray, optional): Optional true feature matrix for input file A. Defaults to None.
        REAL_B (ndarray, optional): Optional true feature matrix for input file B. Defaults to None.
        ngen (int, optional): Number of generations for the genetic algorithm. Defaults to 1000.
        npop (int, optional): Size of the population for the genetic algorithm. Defaults to 200.
        numN (int, optional): Number of parallel processes to use. Defaults to 6.
    """

    random.seed(64)
    
    # Create the pipes for inter-process communication
    pipes = [Pipe(False) for _ in range(numN)]#soal kon az m
    pipes_in = deque(p[0] for p in pipes)
    pipes_out = deque(p[1] for p in pipes)
    pipes_in.rotate(1)
    pipes_out.rotate(-1)
    
    # Define the output files
    out_files = ['/output_' + str(i) for i in range(numN)]
    
    if __name__ == '__main__':
        
        # Set the processes
        processes=[]
        result1_list = []
        result2_list = []
        result3_list = []
        events = []        
        queue = Queue() 
        e = Event()
        events = [Event() for _ in range(numN)]
        processes = [Process(target=mmfs.ivfs, args=(i, inputfile, ngen, ipipe, opipe, e,
                                                     events[i], out_files[i],outputdir,
                                                     real_A,real_B, queue, random.random()))
                     for i, (ipipe, opipe) in enumerate(zip(pipes_in, pipes_out))]
        
        # Start the processes
        for proc in processes:
            proc.start()
            
        # Wait for the events to be set            
        for e in events:
            e.wait()
        
        # Get the results from the queue        
        while not queue.empty():        
            result1, result2, result3 = queue.get()
            result1_list.append(result1)
            result2_list.append(result2)
            result3_list.append(result3)
        
        print('All processes and sub-processes are done!')
        # Compute the final result
        [Best_Solution,Feature_size]=final_result(result1_list,result3_list,inputfile)
        print("The final solution is:",Best_Solution)
        print("The final Feature size is:",Feature_size)
        [Best_Solution_Tr,Feature_size_Tr]=final_result(result2_list,result3_list,inputfile)
        print("The final solution is:",Best_Solution_Tr)
        print("The final Feature size is:",Feature_size_Tr)

###############################################################################
"""                      The Path on Input and Output                       """    
###############################################################################
# The path to the input data CSV file
inputfiles = ['../DATA_500/Data_1.csv',
              '../DATA_500/Data_2.csv']
# The path to the output directory where the MMFSGA results will be saved
outputdirs = ['../1/Data_A_1',
              '../1/Data_A_2',
              '../1/Last_ANS']

###############################################################################
"""       Optional step to check the F1-score between 
          the selected features and the true features                       """    
###############################################################################              
try:
    # Import the true feature matrices from CSV files
    real_A = genfromtxt('Real_A.csv', delimiter=',')
    real_B = genfromtxt('Real_B.csv', delimiter=',')
except FileNotFoundError:
    print("Real_A.csv or Real_B.csv file not found.")
    real_A = None
    real_B = None
print("running from the beginning")             

alldata=[]
for inp, out in zip(inputfiles,outputdirs):
    alldata.append(load_data(inp)["dat"])
    if not os.path.isdir(out):
        os.mkdir(out)

mmfsga(alldata, out, real_A, real_B)


"""
Created on Thu Jul  7 18:10:46 2022
@author: Vandad Imami

This script runs the MMFS-GA algorithm to optimize a Multimodal feature selection problem
using linear classifier. 
Linear Discriminant Analysis (LDA) for binary classification and Multinomial Logistic regression
for multicalss classification.
# This script loads each dataset from CSV files, where the path to the data files need to be specified by inputfiles variable.
"""
import random, os, pickle
import numpy as np
import pandas as pd
import mmfs2GA as mmfs
from multiprocessing import Event, Pipe, Process, Queue
from collections import deque
from numpy import genfromtxt
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
# from sklearn import model_selection 
# from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
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
###############################################################################
def selected_solution(alldata,all_view_solution,best_solutions):
    """
    Evaluates all solutions and returns the best solution and selected features based on accuracy.
    
    Args:
    - dat (list): A list of tuples, each containing the training data and class labels 
    - all_view_solution (list): A list of arrays all possible feature for each view
    - best_solutions (list): A list of arrays representing the selected features for all views
    
    
    Returns:
    - best_solution (tuple): A list representing the best solution, which is an array of binary values
    
    """
    # Iterate through all solutions and evaluate them using correct classifier
    accuracy=[]
    selected_features=[]
    for i in range(len(best_solutions)):
        solution = best_solutions[i]
        [X,y,se_fe]=mmfs.make_data(alldata,all_view_solution,solution)
        selected_features.append(se_fe)
        accuracy.append(mmfs.evalfsolution(X, y))
    [best_solution, best_accuracy]=mmfs.findBestAccuracy(selected_features, accuracy)    
    np.savetxt('SelectedFeatures.csv', best_solution)
    return best_solution
                             
###############################################################################
"""                      Define MMFS-GA Algorithm                           """    
###############################################################################     
def mmfsga(inputfile, outputdir, allviewbest,real_A=None, real_B=None,method=None,view=0, ngen=1000, npop=200, numN=6):
    """
    This script runs the 'ivfs' function from the 'mmfs2GA' module on multiple processes,
    using a set of input parameters and input files.
    Parameters:
        inputfile (list): data set as a list of tuple (Each tuple is one view)
        outputdir (str): Path to the output directory where the MMFSGA results will be saved
        REAL_A (ndarray, optional): Optional true feature matrix for input file A. Defaults to None.
        REAL_B (ndarray, optional): Optional true feature matrix for input file B. Defaults to None.
        ngen (int, optional): Number of generations for the genetic algorithm. Defaults to 1000.
        npop (int, optional): Size of the population for the genetic algorithm. Defaults to 200.
        numN (int, optional): Number of parallel processes to use. Defaults to 6.
    """

    random.seed(64)
    
    # Create the pipes for inter-process communication
    pipes = [Pipe(False) for _ in range(numN)]
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
        events = []        
        queue = Queue() 
        e = Event()
        events = [Event() for _ in range(numN)]
        if method=='ivfs':
            processes = [Process(target=mmfs.ivfs, args=(i, inputfile, ngen, ipipe, opipe, e,
                                                         events[i], out_files[i],outputdir,
                                                         real_A,real_B,view, queue, random.random()))
                         for i, (ipipe, opipe) in enumerate(zip(pipes_in, pipes_out))]
        elif method=='bvfs':
            processes = [Process(target=mmfs.bvfs, args=(allviewbest, inputfile, ngen, npop,
                                                         out_files[i], outputdir, i, ipipe, opipe,
                                                         e, events[i], queue, random.random()))
                         for i, (ipipe, opipe) in enumerate(zip(pipes_in, pipes_out))]
        else:
            print("Error: method is not defined.")
                
                    
        
        # Start the processes
        for proc in processes:
            proc.start()
            
        # Wait for the events to be set            
        for e in events:
            e.wait()
        
        # Get the results from the queue        
        while not queue.empty():        
            result1= queue.get()
            print("Results=",result1)
            result1_list.append(np.array(result1))            
        print('All processes and sub-processes are done!',len(result1_list))
        return result1_list               

###############################################################################
"""                      The Path on Input and Output                       """    
###############################################################################


inputfiles = ['/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/DATA_500/New_Data_A.csv',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/DATA_500/New_Data_B.csv',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/DATA_500/New_Data_ChiSq_Noise.csv',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/DATA_500/New_Data_Normal_Noise.csv',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/DATA_500/New_Data_Uniform_Noise.csv']

outputdirs = ['/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/Data_A_1',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/Data_B_1',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/Data_C_1',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/Data_D_1',
              '/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/Last_ANS']


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

###############################################################################
"""                         Run the MMFS-GA                                 """    
###############################################################################   
alldata=[]
for inp, out in zip(inputfiles,outputdirs):
    alldata.append(load_data(inp)["dat"])
    if not os.path.isdir(out):
        os.mkdir(out)
# Path to save solutions of each view
allsolutionpath='/home/users/vandadim/Desktop/ab/GeneticAlgorithm/check_sampo/Data_D_1/'
os.makedirs(allsolutionpath,exist_ok=True)
all_view_solution=[]
for i,data in enumerate (alldata):
    view_solution=mmfsga(data, out,all_view_solution, real_A, real_B,method='ivfs',view=i)
    if view_solution is not None:        
        solutions=[]
        for d, sol in enumerate(view_solution):
            if d==0:
                solution=np.concatenate(view_solution[0],axis=0)
            else:
                solution=np.array(view_solution[d][0,1:3,:])
            
            solutions.append(np.array(solution))
        concatenated_solutions = np.concatenate(solutions, axis=0)         
        all_view_solution.append(concatenated_solutions)
sn1=['all_Solution.p']
for obj , nameadd in zip([all_view_solution],sn1):
    R = open(allsolutionpath+nameadd, 'wb')                            
    pickle.dump(obj, R)
    R.close()
################################################################
#################################################################
""" REMOVETHIS"""
##################################################################
for i in range (2):
    view_1=all_view_solution[i]
    for j in range(len(view_1)):
        s=view_1[j]
        if i==0:
            print("View_1 f1_score is=",f1_score(real_A,s))
        if i==1:
            print("View_2 f1_score is=",f1_score(real_B,s))

#################################################################
""" REMOVETHIS"""
##################################################################
best_solutions=mmfsga(alldata, out,all_view_solution, real_A, real_B,method='bvfs',view=i)

# Compute the final result
Best_Solution=selected_solution(alldata,all_view_solution,best_solutions)
# print("The final solution is:",Best_Solution)
# print("The final Feature size is:",Feature_size)
# [Best_Solution_Tr,Feature_size_Tr]=final_result(result2_list,result3_list,inputfile)
# print("The final solution is:",Best_Solution_Tr)
# print("The final Feature size is:",Feature_size_Tr)




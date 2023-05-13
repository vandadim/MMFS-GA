"""
Created on Wed Sep 16 14:46:24 2020
@author: Vandad Imani

This module contains two functions: 'bvfs' and 'ivfs'. The 'bvfs' function implements 
a multi-objective genetic algorithm for multiview feature selection and returns 
the best solution found. 
The 'ivfs' function applies the genetic algorithm to multiple datasets and returns the best solution found for each dataset.
The 'bvfs' function selecting the optimal set of modalities from multi-view data, referred to as between-view feature
selection (BV-FS).
"""


from sklearn import model_selection 
from deap import base, creator, tools, algorithms
import random, pickle, fcntl, os
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score#confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression



#################################################################################################################
def generateBit(ps,pu):
    """
    Generate a random number from the set [0, 1], using the probabilities.
    pu is the probability of selecting 0 and ps  is the probability of selecting 1
     """
    return int(np.random.choice([0, 1], size=(1,), p=[pu, ps])[0])
#################################################################################################################
# Here we generate Bits for pops based on each data view  
#################################################################################################################
def generateIntBit():
    return int(np.random.randint(0,3))
#################################################################################################################

#################################################################################################################
def migPipe(deme, k, pipein, pipeout, selection, replacement=None):
    """
    This function migrates individuals from one population to another.

    Args:
    deme: A list of individuals to migrate from.
    k: An integer representing the number of individuals to migrate.
    pipein: A Pipe object used for receiving individuals from another population.
    pipeout: A Pipe object used for sending individuals to another population.
    selection: A function used to select the individuals to migrate.
    replacement: A function used to select the individuals to be replaced in the receiving population. 
                 If None, the migrants will replace individuals in the receiving population.

    Returns:
    None.

    """
    # Select emigrants
    emigrants = selection(deme, k)
    
    if replacement is None:
        # If no replacement strategy is selected, replace those who migrate
        immigrants = emigrants
    else:
        # Else select those who will be replaced
        immigrants = replacement(deme, k)
        
    # Set the file descriptor of pipein to non-blocking mode    
    fd = pipein.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    
    # Send emigrants to another population and receive immigrants from it
    pipeout.send(emigrants)
    buf = None
    
    while buf is None:
        try:
            buf = pipein.recv()
        except IOError:
            # ignore and try again
            pass    
    # Replace the individuals in the receiving population with the immigrants
    for place, immigrant in zip(immigrants, buf):
        indx = deme.index(place)
        deme[indx] = immigrant
#################################################################################################################  

################################################################################################################# 
def make_data(dat,best_pop,individual):
    """
    Selects the best features from input data based on the given population
    and individual using genetic algorithm.
    
    Args:
    - dat: A list of tuples, where each tuple contains the input features and output.
    - best_pop: A list of lists, where each sublist represents a population and contains binary chromosomes.
    - individual: A list of integers, where each integer represents the index of the best chromosome in the corresponding population.
    
    Returns:
    A tuple containing the selected features, output, and the total number of selected features.
    """
    
    # Concatenate all views.
    all_data= np.concatenate([dt[0] for dt in dat],axis=1) 
    # FIND Chromosomes according to the population
    chromosome_mask=[]
    for i in range(len(dat)):
        chromosome_mask=np.append(chromosome_mask,best_pop[i][individual[i]],axis=0) 
    
    # Find the indices of the selected features.
    con_ind = [i for (i, b) in enumerate(chromosome_mask) if b == 1] 
    
    # Form new_data according to slected features
    new_data = all_data[:,con_ind]    
    label = dat[0][1] # True labels
    
    return (new_data, label,sum(chromosome_mask))

#################################################################################################################        

def check_pop(population_a, all_pop, dat,nf, nb):
    """
   This function takes in a population of chromosomes, the entire population, training data, the number of folds for 
   cross-validation, and the number of iterations for the LDA model. It uses LDA to calculate the accuracy of each 
   chromosome and returns the best chromosome and its accuracy.
   
   Args:
       population_a (list): A list of binary chromosomes representing the selected features.
       all_pop (list of lists): A list of lists, where each sublist represents a population and contains binary chromosomes.
       Train_data (list of tuples): A list of tuples containing the training input features and output.
       nf (int): An integer representing the number of folds for cross-validation.
       nb (int): An integer representing the number of iterations for the LDA model.
   
   Returns:
       A tuple containing the best chromosome and its accuracy.
   """
    population_a=np.array(population_a) 
    # Concatenate all data
    all_data = np.concatenate([dt[0] for dt in dat],axis=1)        
    label = dat[0][1]    
    init_acc=0
    
    for ind in range (len(population_a)):
        solution = population_a[ind]
        if sum(solution) == 0:
            continue            
        chromosome_mask=[]
        for i in range(len(solution)):
            chromosome_mask=np.append(chromosome_mask,all_pop[i][solution[i]],axis=0)
        con_ind = [i for (i, b) in enumerate(chromosome_mask) if b == 1]
        if sum(con_ind) == 0:
            continue
        new_data = all_data[:,con_ind]        
        mean_scores = []
        for param in range(nb):
            scores = []
            # normal cross-validation
            kfolds = model_selection.StratifiedKFold(shuffle=True, random_state=None)            
            for train_index, test_index in kfolds.split(new_data,label):
                
                # split the training data
                X_train, X_test = new_data[train_index],new_data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                # set and fit random forest model
                cl_rf = LinearDiscriminantAnalysis()
                cl_rf.fit(X_train, y_train)
                y_pred=cl_rf.predict(X_test)
                scores.append(balanced_accuracy_score(y_test, y_pred))
                
            mean_scores.append(np.mean(scores))
        # get maximum score value    
        value = np.mean(mean_scores)
        
        if value > init_acc:
            best_ind = solution
            best_ACC = value   
    
    return best_ind, best_ACC  

#################################################################################################################        
############# Evaluation function for the IV-FS #######################
#################################################################################################################
def evalfunction(individual, data, nf, nb):   
    """
    This function evaluates the fitness of an individual in a genetic algorithm.

    Args:
       individual (list): A binary vector representing an individual.
       data (tuple): A tuple containing the input features (X) and the target variable (y) of the dataset.
       nf (int): The number of folds for cross-validation.
       nb (int): The number of times to run cross-validation.

    Returns:
       A tuple of two values: the fitness value and the sum of the elements in the individual.
   """
    # Check if the individual has no selected features
    if sum(individual) == 0:
        # Return zero values for fitness and sum
        return 0, 0
    # Get the indices of the selected features
    con_ind = [i for (i, b) in enumerate(individual) if b == 1]
    X = data[0][:,con_ind]
    y = data[1]
    
    mean_scores = []
    for param in range(nb):
        scores = []
        # normal cross-validation
        kfolds = model_selection.StratifiedKFold(shuffle=True, random_state=None)        
        for train_index, test_index in kfolds.split(X=X,y=data[1]):
            
            # Split the training and testing data for the current fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Create a Linear Discriminant Analysis (LDA) model and fit it to the training data
            cl_rf = LinearDiscriminantAnalysis()
        
            cl_rf.fit(X_train, y_train)
            y_pred=cl_rf.predict(X_test)
            scores.append(balanced_accuracy_score(y_test, y_pred))
        
        # calculate mean score for folds
        mean_scores.append(np.mean(scores))
        
    # Compute the mean score for the nb runs of cross-validation    
    value = np.mean(mean_scores)
    
    # Return the fitness value and the number of selected features
    return value, sum(individual)
#################################################################################################################

############# Evaluation function for the Multichromosome #######################

#################################################################################################################
def evalfunctionGA2(individual, allPopulations, data, nf, nb):    #Vandad
    """
    Evaluate the fitness of a genetic algorithm individual by performing linear discriminant analysis
    on a subset of features selected by the individual and calculating the balanced accuracy score
    using k-fold cross-validation.

    Args:
    - individual (list): A list of binary values indicating which features to select.
    - allPopulations (list): A list of all individuals in the genetic algorithm population.
    - data (numpy array): An array of data points to use in the analysis.
    - nf (int): The number of folds to use in k-fold cross-validation.
    - nb (int): The number of times to repeat the analysis.

    Returns:
    - value (float): The fitness value of the individual, calculated as the mean balanced accuracy score
      over nb repetitions of k-fold cross-validation.
    - Sum_chr (int): The sum of the values in the individual.
    """
    
    # Check if the sum of individual values is zero
    if sum(individual) == 0:        
        return 0,0
    # Create a subset of the data using the selected features in the individual
    [X,y,Sum_chr]=make_data(data, allPopulations, individual)
    if X.shape[1]==0:
        print("Warning!! There is no feature to select")
        return 0,0
    
    else:
        mean_scores = []
        for param in range(nb):
            scores = []
            
            # normal cross-validation
            kfolds = model_selection.StratifiedKFold(n_splits=nf, 
                                                  shuffle=True, 
                                                  random_state=None)
            for train_index, test_index in kfolds.split(X=X,y=y):
              
                # split the training data
                X_train, X_test = X[train_index], X[test_index]            
                y_train, y_test = y[train_index], y[test_index]
            
            
            # set and fit random forest model
                cl_rf = LinearDiscriminantAnalysis()
                cl_rf.fit(X_train, y_train)
                y_pred=cl_rf.predict(X_test)
                scores.append(balanced_accuracy_score(y_test, y_pred))
                   
            mean_scores.append(np.mean(scores))
        
    # get maximum score value    
        value = np.mean(mean_scores)
        
        return value, Sum_chr
#################################################################################################################
######################### POPULATION EVALUATION(BASED ON REPEATED CHROMOSOME) ###################################
#################################################################################################################    
def findMostRepeated(A):
    A=np.array(A)
    SUM_Value=np.sum(A,axis=0)
    tmp = [] 
    for i in range(SUM_Value.shape[0]):
        if SUM_Value[i] >= len(A)/2:
            tmp.append(1)
        else:
            tmp.append(0)
    return tmp
#################################################################################################################
################################# POPULATION EVALUATION(BASED ON ACCURACY) ######################################
#################################################################################################################    
def findBestAccuracy(A, acc):
    A=np.array(A)
    acc=np.array(acc)
    idx = np.argmax(acc)
    return A[idx], acc[idx]
#################################################################################################################
#################################################################################################################    
def jaccard2(st1, st2, feat_names):
    """
    Compute the Jaccard similarity coefficient between two sets of features.

    Args:
    st1: a list of 0s and 1s representing the presence or absence of features for the first set.
    st2: a list of 0s and 1s representing the presence or absence of features for the second set.
    feat_names: a list of strings representing the names of the features.

    Returns:
    The Jaccard similarity coefficient between the two sets of features.
    """
    # Convert the feature lists to sets
    st1 = [feat_names[i] for (i, b) in enumerate(st1) if b == 1]
    st2 = [feat_names[i] for (i, b) in enumerate(st2) if b == 1]
    st1 = set(st1)
    st2 = set(st2)
    
    # Compute the Jaccard similarity coefficient
    union = st1.union(st2)
    inter = st1.intersection(st2)
    if len(inter) == 0: 
        jaccard_res = 0
    else:
        jaccard_res = (float(len(inter))/float(len(union)))
    return jaccard_res
#################################################################################################################        

#################################################################################################################        
def getSimScore2(population, feat_names):
    """
    Compute the average Jaccard similarity coefficient between all pairs of individuals in a population.

    Args:
    population: a list of individuals, where each individual is represented as a list of 0s and 1s indicating the presence or absence of features.
    feat_names: a list of strings representing the names of the features.

    Returns:
    The average Jaccard similarity coefficient between all pairs of individuals in the population.
    """
    
    sum_sim_score = 0
    ncomp = 0
    
    # Compute the Jaccard similarity coefficient between all pairs of individuals
    for i in range(len(population)):
        for j in range(len(population)):
            if i > j:
                sum_sim_score = sum_sim_score + jaccard2(population[i],
                                                         population[j],
                                                         feat_names)
                ncomp += 1
    # Compute the average similarity score
    avg_sim_score=float(sum_sim_score/ncomp)
    return avg_sim_score
#################################################################################################################        
#################################################################################################################        
def percentage(percent, whole):
    return (percent * whole) / 100.0
#################################################################################################################
################## GA using multichromosme version for multi view ##################################
################################################################################################################# 
def bvfs(all_Populations, dat, ngen, npop, out_file, Path, procid, pipein, pipeout, sync, seed=None):
    """
    This function performs a bi-objective optimization using NSGA-II algorithm and multi-Gaussian naive Bayes 
    classifier for feature selection. The function returns the best chromosome, accuracy, and the number of 
    selected features on the validation and training set. 

    Parameters
    ----------
    all_Populations : list
        A list of population of chromosomes.
    dat : numpy.ndarray
        The input dataset.
    ngen : int
        The number of generations.
    npop : int
        The size of population.
    out_file : str
        The name of the output file.
    Path : str
        The path of the output directory.
    procid : int
        The ID of the process.
    pipein : multiprocessing.connection.Connection
        The input pipe for migration.
    pipeout : multiprocessing.connection.Connection
        The output pipe for migration.
    sync : multiprocessing.synchronize.Event
        The synchronization event for logging.
    seed : int, optional
        The random seed for generating random numbers. 

    Returns
    -------
    Best_com : list
        The best chromosome selected by the algorithm.
    Best_acc : float
        The accuracy of the best chromosome on the validation set.
    Best_com_tr : list
        The best chromosome selected by the algorithm on the training set.
    Best_acc_tr : float
        The accuracy of the best chromosome on the training set.
    all_Populations : list
        The updated list of population of chromosomes.

    """
    # Create a directory for Second_GA 
    Second_GA = Path + 'Second_GA' + '/'    
    os.makedirs(Second_GA,exist_ok=True)
          
    random.seed(seed)
      
     
    # Create the fitness function with two objectives   
    creator.create("FitnessMulti_5", base.Fitness, weights=(1.0, -1.0))
    
    # Create an individual chromosome 
    creator.create("Individual5", list, fitness=creator.FitnessMulti_5)
    toolbox = base.Toolbox()

    # Register a function to generate an integer chromosome    
    toolbox.register("genIntBit",generateIntBit)
    
    
    # Set the migration rate
    mig_rate = 50
    
    # Set the size of population and number of generations
    npop=50
    ngen=100
    
    # Set the minimum accuracy
    min_acc= 0
    population=[]
    # Set the percentage of best individuals for migration
    best_k = int(percentage(25, npop))    
    toolbox.register("individual2", tools.initRepeat, creator.Individual5, toolbox.genIntBit, n = len(dat))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual2, n = npop)    
    # Register the fitness function to evaluate an individual chromosome
    toolbox.register("evaluate", evalfunctionGA2, allPopulations = all_Populations, data = dat,  nf = 3, nb = 1)    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("migrate", migPipe, k=best_k, pipein=pipein, pipeout=pipeout, selection=tools.selBest, replacement=random.sample)
    
    # Init the population of indiduals    
    population = toolbox.population()
    
    # Evaluate the fitness of the initial population
    fits = toolbox.map(toolbox.evaluate, population)
    
    # Accuracy of all individuals
    acc=[]
    for fit, ind in zip(fits, population):         
        ind.fitness.values = fit 
        acc.append(fit[0])
    
    # Compute the mean accuracy of the initial population
    crt_mean_fit = sum(acc)/len(population)
    
    # Init the logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'procid', 'fv', 'ft', 'M_F', 'avg', 'max_acc']
    
    # Update the logbook    
    logbook.record(gen=0, procid=procid, fv=0, ft=0, M_F=0, avg=0,MAX_ACC=0)
    
    if procid == 0:
        # Synchronization needed to log header on top and only once
        print(logbook.stream)
        sync.set()
    else:
        logbook.log_header = False  # Never output the header
        sync.wait()
        print(logbook.stream)
    
    # Begin the generational process
    for gen in range(1, ngen + 1):        
        
        # Update the univariate rank
        offspring = algorithms.varOr(population, toolbox, lambda_=100, cxpb=0.5, mutpb=0.1)                      
        fits = toolbox.map(toolbox.evaluate, offspring)
        
        acc=[]
        fea=[]
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit            
            acc.append(fit[0])
            fea.append(fit[1])
        
        population = toolbox.select(population + offspring, k = npop)         
        
        accpop_I = [ind.fitness.values[0] for ind in population]
      #  
        if max(accpop_I)==0:
            fv=0
        else:
            fv = (max(accpop_I) - (sum(accpop_I) / len(population))) / max(accpop_I)#
        #
        ft = (sum(accpop_I) / len(population)) - crt_mean_fit#dif avg between the current one gen
        
       # 
        Num_F = np.mean(fea)
        macc = np.mean(accpop_I)
        
        
        maxpopp = max(accpop_I)

        # Update current mean fitness
        crt_mean_fit = sum(accpop_I)/len(population)

        
        
        # Append the current generation statistics to the logbook
        logbook.record(gen = gen, procid=procid, fv = fv, ft = ft, M_F=Num_F, avg = macc, MAX_ACC=maxpopp)
        
        AA= logbook.stream
        with open(Second_GA+out_file + "logbook.txt", "a") as ff:
            ff.write(AA + "\n")
        print(AA) 
        
     
        if maxpopp > min_acc:
            Main_Population_I = population
            Main_ACC_I=accpop_I
            min_acc = maxpopp
            
        if gen % mig_rate == 0 and gen > 0:
            toolbox.migrate(population)
        
        
    #            
    MAIN_OUT = Second_GA + '/'
    
    os.makedirs(MAIN_OUT,exist_ok=True)
    #    
    sn=['View_Population.p','View_ACC.p','all_Populations.p']
    #
    for obj , nameadd in zip([Main_Population_I, Main_ACC_I, all_Populations],sn):
        
        RC = open(MAIN_OUT+out_file+nameadd, 'wb')
        pickle.dump(obj, RC)
        RC.close()
    #
    [Best_com, Best_acc]=findBestAccuracy(Main_Population_I, Main_ACC_I)
    [Best_com_tr, Best_acc_tr]= check_pop(Main_Population_I, all_Populations, dat,nf=10, nb=10)
    #
    return Best_com,Best_acc,Best_com_tr,Best_acc_tr,all_Populations

#################################################################################################################        
#################################################################################################################        
              
def ivfs(procid, dat, ngen, pipein, pipeout, sync, ev,
          out_file, output_dir, real_A, real_B, queue, seed=None):
    """
   This function implements the intra-view feature selection (IV-FS) algorithm 
   for feature selection for each view.

   Args:
   - procid (int): process id number
   - dat (list): data set as a list of tuple (Each tuple is one view)
   - ngen (int): number of generations
   - pipein (multiprocessing.Pipe): input pipe for inter-process communication
   - pipeout (multiprocessing.Pipe): output pipe for inter-process communication
   - sync (multiprocessing.Event): event used for synchronization between processes
   - ev (multiprocessing.Event): event used for external control of the function
   - out_file (str): name of the output file
   - output_dir (str): path of the output directory
   - real_A (int): maximum number of features allowed in a solution
   - real_B (int): minimum number of features allowed in a solution
   - queue (multiprocessing.Queue): queue for inter-process communication
   - seed (int, optional): seed for random number generation. Defaults to None.

   Returns:
   - None
   """
    
    best_pop =[]
    Main_out = output_dir + '/'
    First_ga = Main_out + '/First_GA' + '/'    
    os.makedirs(First_ga,exist_ok=True)
    
        
    for i,data in enumerate (dat):
        
        """
        Determine npop, Ps, and Pu based on data set size
        Args:
        npop (int): number of individuals in population
        ps (float): probability of selecting features
        pu (float): 1-ps
        """
        if data[0].shape[1] <= 100:
            npop=100
            ps=0.9
            pu=0.1
        else:
            npop=200
            ps=0.8
            pu=0.2
        gen_rate = ngen # end of generation
        ssc_10=[]  # Initialize an empty list to store the similarity score     
        flag=False # Initialize the duplicate solution flag
        max_feature= data[0].shape[1] # Initialize the maximum number of features
        minimum_accuracy= 0 # Initialize the minimum amount of accuracy
        good_population = [] # Initialize an empty list to store the good population of individuals
        population = [] # Initialize an empty list to store a population of individuals.
        save_interval = int(ngen * 0.3) # Save every 30%, 60%, or 90% of total generations
        interval_individual=[] # Initialize an empty list to store the good individuals during interval
        best_inds= [] # Initialize an empty list to store the best individuals at the end of generation
        best_inds.append(np.array(np.zeros(data[0].shape[1]))) # Initialize a zero vector
        accpop=[]
        # Create the classes and tools for the genetic algorithm    
        creator.create("FitnessMulti_"+str(i), base.Fitness, weights=(1.0, -1.0))
        all_attributes=vars(creator)
        creator.create("Individual"+str(i), list, fitness=all_attributes["FitnessMulti_"+str(i)])
        all_attributes=vars(creator)
        toolbox = base.Toolbox()
        # The genbit function calls the generateBit function with the arguments ps and pu
        toolbox.register("genbit", generateBit,ps,pu)  
        
        random.seed(seed)
        mig_rate = 50
        best_k = int(percentage(25, npop))
        toolbox.register("individual", tools.initRepeat, all_attributes["Individual"+str(i)], toolbox.genbit, n = data[0].shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = npop)
        toolbox.register("evaluate", evalfunction, data = data,  nf = 10, nb = 10)
        toolbox.register("mate", tools.cxUniform, indpb=0.3)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.9)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("migrate", migPipe, k=best_k, pipein=pipein, pipeout=pipeout,
                         selection=tools.selBest, replacement=random.sample)
        toolbox.register("get_ssc", getSimScore2, feat_names = data[2])
        toolbox.register("mutateSIM", tools.mutShuffleIndexes, indpb=1)
        toolbox.register("CXmate", tools.cxTwoPoint)
        
        
        
        print("################# DATA_VIEW_"+str(i)+"#################",(data[0].shape))
        Save_data = First_ga + 'Data_' + str(i) + '/'
        os.makedirs(Save_data,exist_ok=True)
        
        # Initialize a population
        population = toolbox.population()  
        
        # Evaluate the fitness of each individual in the population
        # and store the results in the 'fits' list
        fits = toolbox.map(toolbox.evaluate, population)
        
        # Update the fitness values of each individual in the population 
        # with their corresponding fitness evaluation from 'fits'
        # and append the accuracy to the 'acc' list
        acc=[]                
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
            acc.append(fit[0])
            
        # Calculate the mean fitness of the population 
        crt_mean_fit = sum(acc)/len(population)
        
        # Init the logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'procid', 'fv', 'ft', 'fe_number', 'ssc','avg','max_acc']
        # Update the logbook
        logbook.record(gen=0, procid=procid, fv=0, ft=0, fe_number=0, ssc=0, avg=0, max_acc=0)
              
        if procid == 0:
            # Synchronization needed to log header on top and only once
            print(logbook.stream)
            sync.set()
        else:
            logbook.log_header = False  # Never output the header
            sync.wait()
            print(logbook.stream)
        
        # Begin the generational process            
        for gen in range(1, ngen + 1):
            # Update the univariate rank # probability cross over and mutatuion must be less than 1
            offspring = algorithms.varOr(population, toolbox, lambda_=100, cxpb=0.2, mutpb=0.1)
            if flag==True:
                flag=False
                cxb=0.9
                mutb=0.9                          
                offspring = [toolbox.clone(ind) for ind in population]
                # Apply crossover and mutation on the offspring
                for bi in range(1, len(offspring), 2):
                    if random.random() < cxb:
                        offspring[bi - 1], offspring[bi] = toolbox.CXmate(offspring[bi - 1], offspring[bi])
                        del offspring[bi - 1].fitness.values, offspring[bi].fitness.values
                for ci in range(len(offspring)):
                    if random.random() < mutb:
                        offspring[ci], = toolbox.mutateSIM(offspring[ci])
                        del offspring[ci].fitness.values
                            
            fits = toolbox.map(toolbox.evaluate, offspring)
            acc=[]
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
                acc.append(fit[0])
                
            # Select new individuals from the offspring and existing population
            # using the NSGA-II selection method    
            population = toolbox.select(offspring + population, k = npop) 
            accpop = [ind.fitness.values[0] for ind in population]
            
            # Calculate the diversity metric fv, defined as the fraction of the
            # difference between the best fitness and the average fitness of 
            # the population, relative to the best fitness
            fv = (max(accpop) - (sum(accpop) / len(population))) / max(accpop)#
            
            # Calculate the difference in average fitness between the current 
            # generation and the previous one
            ft = (sum(accpop) / len(population)) - crt_mean_fit
            
            # Calculate the mean number of features for individuals in the population
            fe_number = np.mean([sum(ind) for ind in population])
            
            # Calculate the similarity score between individuals in the population
            ssc=toolbox.get_ssc(population)
            ssc_10.append(ssc)
                      
            # Check the last 10 similarity scores
            # If so, set the flag to True
            if np.size(ssc_10)==10:
                std_ssc=np.std(ssc_10)
                ssc_10=[]
                if (std_ssc<0.1) and (ssc>0.8):
                    flag=True
                    print("FLAG IS TRUE")
            macc = np.mean(accpop)
            
            # Calculate the mean fitness of the population
            maxpop=max(accpop)               
        
            # Update current mean fitness
            crt_mean_fit = sum(accpop)/len(population)
        
            # Append the current generation statistics to the logbook
            logbook.record(gen = gen, procid = procid, fv = fv, ft = ft,
                           fe_number = fe_number, ssc = ssc, avg = macc, max_acc=maxpop)
            # Get the stream of logbook and append it to a file
            AA= logbook.stream
        
            with open(Save_data+out_file + "logbook.txt", "a") as ff:
                ff.write(AA + "\n")
            print(AA)             
            
                
            # If the maximum accuracy and the number of features are achieved, 
            # update the best population     
            if maxpop >= minimum_accuracy and fe_number < max_feature:
                good_population = population
                Main_ACC =accpop
                minimum_accuracy = maxpop
                max_feature = fe_number
                
            # Send and receive migrants at every 50 generation 
            if gen % mig_rate == 0 and gen > 0:                        
                toolbox.migrate(population)
                
            # Save the results for this iteration at every 30%, 60% and 90% of generation
            if gen % save_interval == 0:
                interval_individual.append(np.array(findBestAccuracy(population,
                                                                     accpop)[0]))
                
            # Collect the best individuals at end of generation    
            if  gen % gen_rate == 0:
                best_inds.append(np.array(findBestAccuracy(good_population, Main_ACC)[0]))
                ACC_BEST=np.array(findBestAccuracy(good_population, Main_ACC)[0])#################
                best_inds.append(np.array(findMostRepeated(good_population)))
                REP_BEST=np.array(findMostRepeated(good_population))
                # Calculate the f1 score between true features and selected features
                # if the real data is available    
                if isinstance (real_A, np.ndarray) and i==0:
                    F_SCORE_ACC=f1_score(real_A, ACC_BEST)
                    F_SCORE_REP=f1_score(real_A, REP_BEST)
                    print("DICE_SCORE_ACC_DATA_A_process_"+str(procid)+"=",F_SCORE_ACC)
                    print("DICE_SCORE_REP_DATA_A_process_"+str(procid)+"=",F_SCORE_REP)
                
                if isinstance (real_B, np.ndarray) and i==1:
                    F_SCORE_ACC=f1_score(real_B, ACC_BEST)
                    F_SCORE_REP=f1_score(real_B, REP_BEST)
                    print("DICE_SCORE_ACC_DATA_B_process_"+str(procid)+"=",F_SCORE_ACC)
                    print("DICE_SCORE_REP_DATA_B_process_"+str(procid)+"=",F_SCORE_REP)
                
                best_inds = np.vstack((best_inds, interval_individual))
                # Save best individuals of each view
                best_pop.append(best_inds)
                    
                sn1=['good_population' + '_Data_' + str(i) + '.p','acc' + '_Data_' + str(i) + '.p']
                for obj , nameadd in zip([good_population, Main_ACC],sn1):
                    R = open(Save_data+out_file+nameadd, 'wb')                            
                    pickle.dump(obj, R)
                    R.close()
                break
               
    
    
    [best_views_pop,best_acc,best_views_pop_Tr,best_acc_Tr,allbest_pop]= bvfs(best_pop, dat, ngen, npop, out_file, Main_out, procid, pipein, pipeout, sync, seed=None)
    print("############_PROC_ID_"+str(procid)+"_############")
    print("BEST_VIEW_POPULATION =",best_views_pop)
    print("BEST_VIEW_ACCURACY =",best_acc)
    print("BEST_VIEW_POPULATION_tr =",best_views_pop_Tr)
    print("BEST_VIEW_ACCURACY_tr =",best_acc_Tr)
    print("##########################################")      
    queue.put((best_views_pop, best_views_pop_Tr,allbest_pop))
    ev.set()
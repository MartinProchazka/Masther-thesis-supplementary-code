import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import random 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
random.seed(42)  
np.random.seed(42)
"""
Script consisting of functions used for postprocessing and dataset generation used in Thesis: Classification in data streams with abrupt concept drift in a subset of features
Some functions assume inputs from script classes.py
"""

def make_example_12_concepts(which_concept, b):
    '''
    Create single sample of the 12 Concepts dataset, for given concept and sum parameter
    
    Args:
        which_concept (int): which concept 0-11
        b (float): parameter for four sum features
    
    Returns:
        list of floats, sample from 12 Concept dataset'''

    #generate label, randomly
    which_label =  random.randint(0,1)

    #generate four sum features
    new_features = [random.randint(0,100) / 10, random.randint(0,100) / 10, random.randint(0,100) / 10, random.randint(0,100) / 10]
    if which_label == 0:
        while new_features[0] + new_features[1] + new_features[2] + new_features[3] <= b:
            new_features = [random.randint(0,100) / 10, random.randint(0,100) / 10, random.randint(0,100) / 10, random.randint(0,100) / 10]
    else:
        while new_features[0] + new_features[1] + new_features[2] + new_features[3]  > b:
            new_features = [random.randint(0,100) / 10, random.randint(0,100) / 10, random.randint(0,100) / 10, random.randint(0,100) / 10]

    #add relevant drifting and irrelevant feature
    if which_concept == 0 : 
        if which_label == 0:
            return [1, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [2, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 1 :
        if which_label == 0:
            return [3, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [1, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 2 :
        if which_label == 0:
            return [4, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [3, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 3 :
        if which_label == 0:
            return [7, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [6, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 4 :
        if which_label == 0:
            return [2, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [11, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 5 :
        if which_label == 0:
            return [4, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [5, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 6 :
        if which_label == 0:
            return [7, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [4, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 7 :
        if which_label == 0:
            return [8, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [9, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 8 :
        if which_label == 0:
            return [3, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [4, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 9 :
        if which_label == 0:
            return [6,  random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [10, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 10 :
        if which_label == 0:
            return [0, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [2, random.randint(0,100) / 10] + new_features, 1
    if which_concept == 11 :
        if which_label == 0:
            return [4, random.randint(0,100) / 10] + new_features, 0
        if which_label == 1:
            return [5, random.randint(0,100) / 10] + new_features, 1
    return False

def create_12_concept_dataset(num_of_example, number_of_concepts, concepts): 
    '''
    Creates 12 Concepts dataset (Subsection 2.2.2), first feature is decisive drifting, second is irrelevant and last four are sum features

    Args:
        num_of_examples (int): number of samples in the dataset
        number_of_concepts (int): number of concepts we want (1-12)
        concepts (floats): defined sum parametrs (list not longer than 12)
    
    Returns:
        tuple of lists, first is list of lists of floats, list of feature vectors, second is list of ints (0,1) which are according labels
    '''

    examples = []
    labels = []
    concept = 0
    for i in range(num_of_example):
        if i %  (num_of_example // number_of_concepts) == 0 and i != 0: #period of concept
            concept += 1
        example, label = make_example_12_concepts(concept,concepts[concept])
        examples.append(example)
        labels.append(label)

    return examples,labels

def create_kolter_dataset(number_of_exmples,concepts,noise=False):
    '''
    Create SEA concepts dataset
    
    Args:
        number_of_exmples (int): number of samples in dataset
        concepts (float): sum parameter
        noise (int or bool): default false (no noise), 1-100 percent number of noise
    
    Returns:
        tuple of two lists, first is list of lists of floats, list of feature vectors, second is list of ints (0,1) which are according labels
    '''

    #noise = % of noisy labels (1-100)
    examples=[]
    labels=[]
    for concept in concepts:
        for _ in range(number_of_exmples//len(concepts)):
            example = [random.randint(0,100),random.randint(0,100),random.randint(0,100)]
            example = [element / 10 for element in example]
            if example[0] + example[1] <= concept:
                label = 0
            else:
                label = 1
            if noise != False :
                if random.randint(1,100) <= noise: 
                    label = (label + 1 ) % 2  #choose another

            examples.append(example)
            labels.append(label)
    return examples,labels

def list_of_correct(num_of_examples,len_of_intervals,prediction,labels,f1 = False):
    '''
    Returns values preprocessed into graph, accuracy in len_of_intervals predictions on y-axis and number of processed samples in x-axis

    Args:
        num_of_examples (int): number of overall comparisons we want to measure accuracy on
        len_of_intervals (int): intervals we divide predictions in and compute accuracy, len_of_intervals should divide len of prediction 
        prediction (list): predictions
        labels (list): gold labels
        f1 (bool): Use F1 score instead
    
    Returns:
        tuple of lists of intes, steps and score
    '''

    y = []
    x = []
    for j in range(num_of_examples // len_of_intervals):
        number_corect = 0
        predict = prediction[j * len_of_intervals : (j + 1) * len_of_intervals]
        label = labels[j * len_of_intervals : (j + 1) * len_of_intervals]
        if f1:
            y.append(f1_score(label,predict))
        else:
            for i in range(len(predict)):
                if label[i] == predict[i]:
                    number_corect += 1
            y.append(number_corect / len(predict))
        x.append(j)
    return x, y

def process_dataset_return_data_for_graph(train_data,hellinger_process,number_of_subsets):

    '''
    Track epsilons and concept drift detection of hellinger_process (class Hellinger_detection) on train_data

    Args:
        train_data (numpy array of lists): feature vectors
        hellinger_process (instance of class Hellinger_detection ): process of Hellinger drift detection
        number_of_subsets (int): number of subsets
    
    Returns:
        tuple of four lists, epsilons and output information about the drift


    '''

    step = 0
    drifted_features = []
    y_epsilons = [[] for _ in range(number_of_subsets)]
    x_epsilons = [[] for _ in range(number_of_subsets)]
    y_drift = [[] for _ in range(number_of_subsets)]
    x_drift = [[] for _ in range(number_of_subsets)]
    for data in train_data:
        step += 1
        hellinger_process.process_sample(data)
        drifted = hellinger_process.get_last_subset_drift()
        epsilons = hellinger_process.get_epsilons()
        for subset in range(len(epsilons)):
            y_epsilons[epsilons[subset][0]].append(epsilons[subset][1])
            x_epsilons[epsilons[subset][0]].append(step)
        if True in drifted:
            for i in range(len(drifted)):
                if drifted[i]:
                    drifted_features.append([i,step])
                    x_drift[i].append(step)
                    y_drift[i].append(epsilons[i][1])

    return x_epsilons, y_epsilons, x_drift, y_drift

def hyperbola_function(x, a, b):
    '''
    Basic hyperbola function
    
    Args:
        a (float): hyperbola parameter
        b (float): hyperbola parameter
        x (float): input to hyperbola
        
    Returns:
        float, value of hyperbola given by params a and b in value x
    '''

    return a* (1 / (b + x)) 

def dirichlet_threshold(alphas, p, num_samples=10**6):
    '''
    Compute Dirichlet threshold value for given parameters
    
    Args:
        alphas (list of floats): determining alphas of Dirichlet distribution
        p (float): between 0 and 1, determining quantile of Dirichlet distribution 

    Returns:
        float, found threshold

    '''
    
    # Generate samples from the Dirichlet distribution
    samples = np.random.dirichlet(alphas, size=num_samples)

    # Find the minimum value for each sample
    min_values = np.min(samples, axis=1)
    
    # Sort the minimum values
    sorted_min_values = np.sort(min_values)
    
    # Find the (1 - p)-th percentile
    threshold_index = int(num_samples * (1 - p))
    threshold = sorted_min_values[threshold_index]
    
    return threshold

def create_prior_threshold(alphas, quantile = 0.95,own_hyperbola_params = False, plot_original_and_curve = False, threshold_for_small_K = [0.15,0.15], return_only_popt = False):
    '''
    Create prior threshold approximated by hyperbola of shape : a * (1 / (x + b))

    Args: 
        alphas (list of lists of floats): determining alphas of Dirichlet distribution, each sublist determine one set of alphas, there can not be sublists of same lenghth
        quantile (float): between 0 and 1, determining quantile of Dirichlet distribution 
        own_hyperbola_params (list of two floats): determining hyperbola : own_hyperbola_params[0] * (1 / (x + own_hyperbola_params[1]))
        threshold_for_small_K (list of two floats): determining values of threshold for number of cluster equal to 0 and 1
        return_only_popt (bool), return fitted hyperbola params
    
    Returns:
        function or list, hyperbola created with found parameters or list of found parameters
     '''
    

    if own_hyperbola_params != False:
        popt = own_hyperbola_params
    else:
        #get quantile values
        quantiles = []
        number_of_alphas = []
        for set_of_alphas in alphas:
            quantiles.append(dirichlet_threshold(set_of_alphas, quantile))
            number_of_alphas.append(len(set_of_alphas))

        popt, _ = curve_fit(hyperbola_function, number_of_alphas, quantiles)
        
        if plot_original_and_curve:
            plt.scatter(number_of_alphas, quantiles, label='Original Data')
            plt.plot(number_of_alphas,[hyperbola_function(number, *popt) for number in number_of_alphas], 'r-', label='Fitted Curve')
            plt.legend()
            plt.show()

        if return_only_popt:
            return popt

    #define hyperbolic threshold
    def found_hyperbola(x):
        if x < 2:
            return threshold_for_small_K[x]
        return popt[0] * (1 / (x + popt[1])) 
    
    return found_hyperbola
    
def main() -> float:
    return 0

if __name__ == "__main__":
    main()
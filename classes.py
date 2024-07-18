import numpy as np
import scipy.stats as stats
import random 
random.seed(42)  
np.random.seed(42)
"""
Script is consisting of classes for computation of key algorithms presented in Thesis: Classification in data streams with abrupt concept drift in a subset of features
- Binary Incremental Gaussian Naive Bayes classifier with possibility of restricted feature vector (Subsection 2.1.2)
- Dynamic Weighted Majority algorithm with possible label delay using Gaussian NB as expert (Algorithm 9) and inbuild following feature subset drift indicators
- Hellinger concept drift detection algorithm  (Algorithm 3)
- Incremental data stream adaptive Gaussian Mixture Model algorithm (Algorithm 4)
For more information see README.txt file
 """

class GMM_overal_process:
    '''
    Algorithm 4
    Overall class for Incremental adaptive GMM algorithm
    '''

    def __init__(self,GMM_params) -> None:
        super().__init__()
        '''
        Create class element, constructor, instance and initialize parameters

        Args:
          GMM_params (list):
            , feature_subsets - list of lists of dependent features (no intersection)
            , t_b             - int, period for mixture  initialization
            , beta            - float, parameter for percentile in outlier detection
            , SP_bounds       - list of two floats, zero position is SP lower bound and on second SP lower bound, both are positive and second is grater than first
            , inc_params      - list of two ints, zero position is needed number of identical prediction for incorporation and on second number of last predictions we track, both are positive and second is grater than first
            , prior_threshold - a function whose input is int and output is float between 0 and 1, calling prior_threshold(input_int)
            , initial_inv_covariance_matrices - lits of positive definit numpy matrices, where i-th matrix correspond to i-th feature subset
        
        '''
        
        feature_subsets, t_b, beta, SP_bounds, inc_params, prior_threshold, initial_inv_covariance_matrices = GMM_params


        #initialize set of mixtures and needed parameters
        self.process_parameters = {"set_of_mixtures" : [ Cluster_mixture(len(subset), t_b, beta, SP_bounds,inc_params, prior_threshold, initial_inv_covariance_matrices[index]) for index, subset in enumerate(feature_subsets) ],
                                   #set of mixtures, one for each feature subset
                                   "feature_subsets" : feature_subsets, #feature subsets determining split of the feature vector X
                                   "dimension" : np.sum([len(subset) for subset in feature_subsets]), #overall number of features
                                   "outlier_prob" : [[0,0,0,0,0,0,0,0,0,0] for _ in feature_subsets], #number of outliers in last 100 samples, initialized to 10 non-outlier due to non-extreme probabilities at the beginning
                                   "processed_samples" : 0,
                                   "SP_upper" : SP_bounds[1]
                                   }

    def process_sample(self,X,classifier):
        '''
        Algorithm 4, process single sample X through Incremental Adaptive GMM alborithm and determine whether it each feature should be used by classifier or not

        Args:
            X (numpy array): consisting single feature vector 
            classifier (object with method ".predict(X,features_suitability)"): for numpy array X and features_suitability, which is list determining whether should be i-th feature used by setting i-it position to True/False, method outputs 0 or 1
 

        Returns: 
            list: filled with True/False determining whether according feature is suitable for classification or not
        '''
        

        #split X according to feature subsets
        X_splitted = [[X[feature] for feature in feature_subset] for feature_subset in self.process_parameters["feature_subsets"]]

        #proces splitted X to mixtures, accorfing to Algorithm 5, get information about fitting of X
        X_fitted = [] #list we keep information about fitt of X into mixtures
        for mixture_index, mixture in enumerate(self.process_parameters["set_of_mixtures"]):
            X_fitted.append(mixture.sample_fit(X_splitted[mixture_index]))

            #update outlier probability
            if X_fitted[-1] == -1:
                self.process_parameters["outlier_prob"][mixture_index].append(1)
            else:
                self.process_parameters["outlier_prob"][mixture_index].append(0)
            if len(self.process_parameters["outlier_prob"][mixture_index]) > 100:
                del self.process_parameters["outlier_prob"][mixture_index][0]


        

        features_suitability = [False] * self.process_parameters["dimension"] #defines whether use according feature for classification, by default "do not use"
        
        #find stable relevant features
        for mixture_index, fitted in enumerate(X_fitted):
            if fitted == 1:
                for feature in self.process_parameters["feature_subsets"][mixture_index]:
                    features_suitability[feature] = True
        
        #determine non-incorporated (fitted == 0) and outlier parts (fitted == -1)
        for mixture_index, fitted in enumerate(X_fitted):
            if fitted == 0:
                features_suitability_non_inc = features_suitability.copy()
                for feature in self.process_parameters["feature_subsets"][mixture_index]:
                    features_suitability_non_inc[feature] = True
                non_incorporated_prediction = classifier.predict(np.array([X]),features_suitability_non_inc)
                stable_relevat_prediction = classifier.predict(np.array([X]),features_suitability)
                if stable_relevat_prediction == non_incorporated_prediction:
                    self.process_parameters["set_of_mixtures"][mixture_index].prediction_comparison_result(1)
                else:
                    self.process_parameters["set_of_mixtures"][mixture_index].prediction_comparison_result(0)
            if fitted == -1:
                #create new cluster using retention set, update retention set paramters
                self.process_parameters["set_of_mixtures"][mixture_index].cluster_creation_procedure(X_splitted[mixture_index],self.process_parameters["processed_samples"],np.sum(self.process_parameters["outlier_prob"][mixture_index]) / len(self.process_parameters["outlier_prob"][mixture_index]))

        #if criterion is met start spurious clusters deletion process (Algorithm 8)
        for mixture in self.process_parameters["set_of_mixtures"]:
            if mixture.get_sum_sp() >= self.process_parameters["SP_upper"]:
                mixture.delete_spurious_clusters()

        self.process_parameters["processed_samples"] += 1

        return features_suitability
    
    def get_set_of_mixture(self):
        '''
        Return current set of mixtures
        
        Returns: 
            list: list of instances class Cluster_mixture, current set of mixtures
        '''

        return self.process_parameters["set_of_mixtures"]
        

class Cluster_mixture:
    '''
    Class standing for mixture of clusters modeling single feature subset
    '''

    def __init__(self,dim, t_b, beta, SP_bounds, inc_params, prior_threshold, initial_inv_covariance_matrix) -> None:
        super().__init__()
        '''
        Create class, constructor and initialize parameters

        Args:  
             initial_inv_covariance_matrix (numpy matrix): positive definit initial inverse of covariance matrix used when there is no cluster in mixture
             dim (int): dimension of samples processed by the mixture
             t_b (int): period for mixture  initialization
             beta (float): parameter for percentile in outlier detection
             SP_bounds (list): has two floats, zero position is SP lower bound and on second SP lower bound, both are positive and second is grater than first
             inc_params (list): has two ints, zero position is needed number of identical prediction for incorporation and on second number of last predictions we track, both are positive and second is grater than first
             prior_threshold (function): whose input is int and output is float between 0 and 1, calling prior_threshold(input_int)

        '''

        #initialize parameters
        self.mixture_parameters = {"dim" : dim,
                                   "t_b" : t_b,  #mixture initializaton period
                                   "beta" : beta,  
                                   "SP_lower" : SP_bounds[0],  #SP lower bound used for forgetting procedure for sp
                                   "SP_upper" : SP_bounds[1],  #SP upper bound used for determining whether should be cluster purification procedure launched or not
                                   "prior_threshold" : prior_threshold, 
                                   "initial_inv_covariance_matrix" : initial_inv_covariance_matrix, 
                                   "mixture" : [],   #list consisting all clusters from this mixture
                                   "d_k" : [],   #list of clusters indexes current sample fits in
                                   "retention_set" : [], #set of outliers in retention set
                                   "ret_set_exp_time" : [], #list of times after which according samples from retention_set will be discarded
                                   "K" : 0, #number of clusters in mixture
                                   "delta" : 0.05, #probability for setting retention expiration parameter
                                   "inc_params" : inc_params
                                   }
    
    def sample_fit(self,X):
        '''
        Algorithm 5, check whether numpy array sample X fits into mixture, update according clusters parameters

        Args:
          X (numpy array): feature vector

        Returns:
          int,   1 if X fits into incorporated clusters
                 0 if X fits into non-incorporated clusters only
                -1 if X is determined as outlier
        '''

        #Compute d_k as set of clusters X fits in
        self.mixture_parameters["d_k"] = []
        for cluster_index, cluster in enumerate(self.mixture_parameters["mixture"]):
            if cluster.sample_distance(X) < stats.chi2.ppf(1 - self.mixture_parameters["beta"], self.mixture_parameters["dim"]):
                self.mixture_parameters["d_k"].append(cluster_index)

        if len(self.mixture_parameters["d_k"]) > 0:
            #there are clusters X fits in
            #update parameters of mixture and determine feature suitability

            sum_N_X = 0    #conditional prior is needed for calculation of responsibility            
            clusters_incorporation = [False] * len(self.mixture_parameters["d_k"])
            for cluster in self.mixture_parameters["mixture"]:
                sum_N_X += cluster.get_prior() * cluster.get_conditional_prior(X)
            for incorporation_index, cluster_index in enumerate(self.mixture_parameters["d_k"]):
                self.mixture_parameters["mixture"][cluster_index].update_params(X,sum_N_X)
                if self.mixture_parameters["mixture"][cluster_index].get_inc(): #False if not incorporated True if it is
                    clusters_incorporation[incorporation_index] = True
            Cluster_mixture.actualize_priors(self)
            if False in clusters_incorporation:
                if True in clusters_incorporation:
                    for cluster_index in self.mixture_parameters["d_k"]:
                        # sample X is already incorporated in some cluster, we consider comparison prediction to be same
                        if self.mixture_parameters["mixture"][cluster_index].get_inc() == False:
                            self.mixture_parameters["mixture"][cluster_index].set_last_value_of_cop_update_inc(1)
                    return 1
                else:
                    #X fits into non-incorporated clusters only, prediction comparison is done in class GMM_overal_process
                    return 0
            else:
                #X is well fitting to incorporated clusters only
                return 1
        else:
            #there is no cluster X fits in, we mark X as an outlier
            return -1
    
    def cluster_creation_procedure(self,X,t,outlier_prob):
        '''
        Algorithm 7, perform cluster creation procedure 
        
        Args: 
            X (numpy array): feature vector
            t  (int): number of processed samples
            outlier_prob (float): prior outlier probability

        Returns: None
        '''


        #using retention set we examine possibility of new cluster creation
        if self.mixture_parameters["K"] > 0:
            #we already have some clusters, we will not use initial covariance matrix
            univ_cov_mat = np.array([np.zeros([self.mixture_parameters["dim"]])])
            for cluster in self.mixture_parameters["mixture"]:
                univ_cov_mat = univ_cov_mat + cluster.get_weighted_variances()
            univ_inv_cov_mat = np.diag([1 / variance for variance in [0.001 if x == 0 else x for x in univ_cov_mat[0]]]) #compute diagonal matrix with inverse variances, if variance is 0 we replace it with constant 0.001

        else:
            #we have to use parameter inv covar matrix
            univ_inv_cov_mat = self.mixture_parameters["initial_inv_covariance_matrix"]

        if t <= self.mixture_parameters["t_b"]:
            #we are still in initialization period
            inc_init = True
        else:
            inc_init = False

        potential_new_cluster = single_cluster(X,univ_inv_cov_mat, inc_init,self.mixture_parameters["inc_params"])
        retention_set_samples = [] #indexes of samples from retention set which creates a new cluster
        for sample_index, retention_sample in enumerate(self.mixture_parameters["retention_set"]):
            if potential_new_cluster.sample_distance(retention_sample) < stats.chi2.ppf(1 - self.mixture_parameters["beta"], self.mixture_parameters["dim"]):
                if len(retention_set_samples) == 0: #we wind first sample from retention set we add to new cluster

                    self.mixture_parameters["mixture"].append(potential_new_cluster)
                    self.mixture_parameters["K"] += 1

                Cluster_mixture.actualize_priors(self)
                
                sum_N_X = 0    #conditional prior is needed for calculation for responsibility            
                for cluster in self.mixture_parameters["mixture"]:
                    sum_N_X += cluster.get_prior() * cluster.get_conditional_prior(X)
                self.mixture_parameters["mixture"][-1].update_params(X,sum_N_X)
                retention_set_samples.append(sample_index)

        if len(retention_set_samples) > 0:
            #we created a valid new cluster, remove used samples from retention set
            for sample_index in sorted(retention_set_samples, reverse = True):
                del self.mixture_parameters["retention_set"][sample_index]
                del self.mixture_parameters["ret_set_exp_time"][sample_index]
        else:
            #we did not created a valid new cluster, add X to retention set
            self.mixture_parameters["retention_set"].append(X)
            self.mixture_parameters["ret_set_exp_time"].append(np.log(self.mixture_parameters["delta"]) / np.log(1 - (1 - outlier_prob) * self.mixture_parameters["prior_threshold"](self.mixture_parameters["K"])))

        #update of retention set expiration parameters
        for sample_index in range(len(self.mixture_parameters["retention_set"]) - 1, -1, -1):
            self.mixture_parameters["ret_set_exp_time"][sample_index] -= 1
            if self.mixture_parameters["ret_set_exp_time"][sample_index] <= 0:
                del self.mixture_parameters["retention_set"][sample_index]
                del self.mixture_parameters["ret_set_exp_time"][sample_index]

    def remove_according_to_LM(self,LM,column_sum_threshold):
        '''
        Algorithm 8, remove clusters according to Logical matrix

        Args:
            LM (numpy matrix): Logical matrix
            column_sum_threshold  (int): threshold for problematic columns of LM

        Returns: 
            numpy matrix, updated LM
        '''
        
        #remove cluster according to LM matrix
        problematic_columns = [index for index, column_sum in enumerate(np.sum(LM,axis=0)) if column_sum >= column_sum_threshold]
        while len(problematic_columns) > 0 and self.mixture_parameters["K"] > 1:
            list_of_priors = [self.mixture_parameters["mixture"][cluster_index].get_sp() for cluster_index in problematic_columns]
            min_prior = problematic_columns[list_of_priors.index(min(list_of_priors))]
            del self.mixture_parameters["mixture"][min_prior]
            self.mixture_parameters["K"] -= 1
            LM = np.delete(LM, min_prior, axis=0) #delete according row
            LM = np.delete(LM, min_prior, axis=1) #delete according column
            deleted_indexes = Cluster_mixture.delete_according_prior_threshold(self)
            for index in deleted_indexes:
                LM = np.delete(LM, index, axis=0)
                LM = np.delete(LM, index, axis=1) 
            problematic_columns = [index for index, column_sum in enumerate(np.sum(LM,axis=0)) if column_sum >= column_sum_threshold]
        
        return LM

    def delete_according_prior_threshold(self):
        '''
        Delete clusters according to prior threshold (delete until there is cluster whose prior probability is under the threshold)

        Returns:
          list of ints, list indexes of deleted cluster, which were deleted in the same order as are in list
        '''

        Cluster_mixture.actualize_priors(self)

        #find spueirous clusters according to prior threshold
        deleted_thresholds = [] 
        spur_clusters=[]
        removed = True
        while removed :
            removed = False
            for cluster_index in range(len(self.mixture_parameters["mixture"]) - 1, -1, -1):
                if self.mixture_parameters["mixture"][cluster_index].get_prior() < self.mixture_parameters["prior_threshold"](self.mixture_parameters["K"]):
                    deleted_thresholds.append(cluster_index)
                    removed = True
        
            #delete found spurious clusters
            for cluster_index in deleted_thresholds:
                self.mixture_parameters["K"] -= 1
                del self.mixture_parameters["mixture"][cluster_index]
                spur_clusters.append(cluster_index)
            
            deleted_thresholds = []

            Cluster_mixture.actualize_priors(self)

        return spur_clusters

    def delete_spurious_clusters(self):
        '''
        Algorithm 8, delete spurious clusters, according to prior threshold and overlay detection, apply fotgetting factor on sp

        Returns: None
        '''

        Cluster_mixture.delete_according_prior_threshold(self)


        #creation of logical matrix, determining clusters overlay
        LM = np.zeros([self.mixture_parameters["K"],self.mixture_parameters["K"]])
        for i, cluster_row in enumerate(self.mixture_parameters["mixture"]):
            for j, cluster_column in enumerate(self.mixture_parameters["mixture"]):
                if cluster_row.sample_distance(cluster_column.get_mean_vector()) <= stats.chi2.ppf(1 - self.mixture_parameters["beta"], self.mixture_parameters["dim"]) and i != j:
                    LM[i,j] = 1

        
        #procedure for columns with sum > 1
        LM = Cluster_mixture.remove_according_to_LM(self,LM,2)

        #now same procedure for columns consisting 1
        Cluster_mixture.remove_according_to_LM(self,LM,1)

        #apply sp forget factor
        sum_sp = Cluster_mixture.get_sum_sp(self)
        factor = np.minimum(1,self.mixture_parameters["SP_lower"] / sum_sp)
        for cluster in self.mixture_parameters["mixture"]:
            cluster.set_sp(factor * cluster.get_sp())
        
        Cluster_mixture.actualize_priors(self)     

    def actualize_priors(self):
        '''
        Actualize priors
        
        Returns: None
        '''

        sp_sum = Cluster_mixture.get_sum_sp(self)
        for cluster in self.mixture_parameters["mixture"]:
            cluster.set_prior(cluster.get_sp() / sp_sum)
    
    def prediction_comparison_result(self,result):
        '''
        Actualize values in incorporation procedure

        Args:
          result (int): 1 if predictions are same, 0 if predictions differ

        Returns: 
            None
        '''

        for cluster_index in self.mixture_parameters["d_k"]:
            self.mixture_parameters["mixture"][cluster_index].set_last_value_of_cop_update_inc(result)

    def get_retention_set(self):
        '''
        Get retention set
                
        Returns: 
            list of numpy arrays, feature vectors from retention set
        '''

        return self.mixture_parameters["retention_set"]

    def get_mixtur(self):
        '''
        Return mixture of clusters
        
        Returns:
            list, of enstances classes single_cluster
        '''

        return self.mixture_parameters["mixture"]
    
    def get_sum_sp(self):
        '''
        Return sum of cluster prior probabilities in the mixture
         
        Returns:
         float, sum of all sp values in the mixture
           '''
        
        sum_sp = 0
        for cluster in self.mixture_parameters["mixture"]:
            sum_sp += cluster.get_sp()
        return sum_sp


class single_cluster:
    '''class representing single cluster used for GMM algorithm'''

    def __init__(self,X, inv_covar_matrix, incorporation_initialization, inc_params) -> None:
        super().__init__()
        '''
        Algorithm 8, create class element, constructor, initialize values in the cluster

        Args:
            X (numpy array): restricted feature vector
            inv_covar_matrix (numpy matrix): positive definit initial inverse of covariance matrix used when there is no cluster in mixture
            incorporation_initialization (bool): True/False, determine whether is cluster suitable for prediction
            inc_params (list of two ints): zero position is needed number of identical prediction for incorporation and on second number of last predictions we track, both are positive and second is grater than first
        
        '''
        #Initialize parameters
        self.cluster_parameters = {'inv_covar_matrix' :inv_covar_matrix,
                                   'dim' : len(X),
                                   'covar_det' : np.prod([1 / inv_covar_matrix[i][i] for i in range(len(X))]),
                                   'mean_vector' : np.array([X]),
                                   'sp' : 1,
                                   'inc' : incorporation_initialization,
                                   'cop' : [0],  #parameter for tracking similar prediction and determining incorporation
                                   'prior' : 0,  #is set and updated by single_cluster.update_prior, directly from cluster mixture
                                   'cop_n_last' : inc_params[1],#41,   #last number of fits needed for incorporation decision
                                   'cop_must_same' : inc_params[0],#35,  #needed number of same prediction out of cop_n_last
                                   #parameters for variance computation:
                                   'number_of_examples' : 1,
                                   'means' : np.sum(X, axis=0),
                                   'var' : np.var(X, axis=0)
                                   }

    def update_params(self,X,sum_N_X):
        '''
        Algorithm 6, update parameters of cluster

        Args:
            X (numpy array): restricted feature vector
            sum_N_X (float): sum of sp valueas from whole mixture of clusters
        
        Returns:
            None

         '''
        responsibility = (self.cluster_parameters['prior'] * single_cluster.get_conditional_prior(self,X)) / sum_N_X
        self.cluster_parameters['sp'] += responsibility
        ratio_poster = responsibility / self.cluster_parameters['sp']
        d_k = single_cluster.sample_distance(self,X)

        #update parameters for variance computation
        self.cluster_parameters["var"] = self.cluster_parameters["var"] + (self.cluster_parameters["number_of_examples"] / (1 + self.cluster_parameters["number_of_examples"])) * ((1 / self.cluster_parameters["number_of_examples"]) * self.cluster_parameters["means"] - X)**2
        self.cluster_parameters["means"] =  self.cluster_parameters["means"] + X
        self.cluster_parameters["number_of_examples"] = self.cluster_parameters["number_of_examples"] + 1

        self.cluster_parameters['inv_covar_matrix'] = (1 / (1 -  ratio_poster)) * (self.cluster_parameters['inv_covar_matrix'] - (ratio_poster * self.cluster_parameters['inv_covar_matrix'] @ (X - self.cluster_parameters['mean_vector']).T @ (X - self.cluster_parameters['mean_vector']) @ self.cluster_parameters['inv_covar_matrix']) / (1 + ratio_poster * d_k))
        self.cluster_parameters['covar_det'] = ((1 - ratio_poster)**self.cluster_parameters['dim'] ) * (1 + ratio_poster * d_k) * self.cluster_parameters['covar_det']
        self.cluster_parameters['mean_vector'] = self.cluster_parameters['mean_vector'] + ratio_poster * (X - self.cluster_parameters['mean_vector'])
        
        #updation of incorporation params takes place in class Cluster_mixture

    def sample_distance(self,X):
        '''
        Compute squared Mahalanobis distance of X form the cluster

        Args:
            X (numpy array): restricted feature vector
        
        Returns:
            float, squared Mahalanobis distance of X form the cluster
        '''

        return (X - self.cluster_parameters['mean_vector']) @ self.cluster_parameters['inv_covar_matrix'] @ (X - self.cluster_parameters['mean_vector']).T

    def get_conditional_prior(self,X):
        '''
        Return conditional prior of the cluster

        Args:
            X (numpy array): restricted feature vector
        
        Returns:
            float, conditional prior of the cluster
        '''
          
        return np.sqrt(1 / ((2 * np.pi)**self.cluster_parameters['dim'] * self.cluster_parameters['covar_det'])) * np.exp(-0.5 * single_cluster.sample_distance(self,X))
    
    def set_prior(self,prior):
        '''
        Set cluster prior

        Args:
            prior (float): new cluster prior
        
        Returns:
            None
        '''

        self.cluster_parameters['prior'] = prior
    
    def get_prior(self):
        '''
        Return prior of the cluster

        Returns:
            float, prior of the cluster
        '''
         
        return self.cluster_parameters['prior']
    
    def get_sp(self):
        '''
        Return sp value of the cluster

        Returns:
            float, sp value of the cluster
        '''

        return self.cluster_parameters['sp']
    
    def set_sp(self,new_sp):
        '''
        Set sp value of the cluster

        Args:
            new_sp (float): new sp value
        
        Returns:
            None
        '''
         
        self.cluster_parameters['sp'] = new_sp

    def get_inc(self):
        '''
        Return incorporation status of the cluster

        Returns:
            int, parameter stating whether cluster is incorporated or not
        '''

        return self.cluster_parameters['inc']
    
    def get_mean_vector(self):
        '''
        Return mean vector of the cluster

        Returns:
            numpy array, mean vector of the cluster
        '''
                
        return self.cluster_parameters['mean_vector']

    def set_inc(self,new_inc):
        '''
        Set incorporation status of the cluster

        Args:
            new_inc (int): new incorporation status
        
        Returns:
            None
        '''

        self.cluster_parameters['inc'] = new_inc

    def get_cop(self):
        '''
        Return number of similar prediction (compared to stable part of the feature vector) of the cluster

        Returns:
            int, number of similar prediction
        '''

        return self.cluster_parameters['cop']
    
    def get_weighted_variances(self):
        '''
        Return weighted variance of the cluster

        Returns:
            float, weighted variance
        '''

        return self.cluster_parameters['prior'] * (1 / self.cluster_parameters["number_of_examples"]) * self.cluster_parameters["var"]
    
    def get_inverse_covar_matrix(self):
        '''
        Return inverse covariance matrix of the cluster

        Returns:
            numpy matrix, inverse covariance matrix
        '''

        return self.cluster_parameters['inv_covar_matrix']

    def set_last_value_of_cop_update_inc(self,new_cop):
        '''
        Update prediction comparisons (cop) of the cluster and stable part of the vector
        
        Args:
            new_cop (int): 1 if predictions agreed, 0 if they disagreed
        
        Returns:
            None

        '''
        if len(self.cluster_parameters['cop']) < self.cluster_parameters['cop_n_last']:
            self.cluster_parameters['cop'].append(new_cop)
        else :
            #we have already 'cop_n_last' comparisons, we drop the oldest one
            self.cluster_parameters['cop'][1:].append(new_cop)

        #check whether incorporation inc changes
        if len(self.cluster_parameters['cop']) >= self.cluster_parameters['cop_must_same'] and np.sum(self.cluster_parameters['cop']) >= self.cluster_parameters['cop_must_same']:
            single_cluster.set_inc(self,True)


class NBModel:
    '''
    Chapter 2, Binary Incremental Naive Bayes classifier with possibility of restricted feature vector classification
    '''
    def __init__(self,train_data,train_target,smooth=False) -> None:
        super().__init__()
        '''
        Constructor, initialize NB parameters
        
        Args:
            train_data (numpy array of lists): feature vectors
            train_target (numpy array of ints): values 1 and 0, labels
            smooth (bool): whether apply smoothing of label prior probability or not
        
        
        '''

        #initialize parameters
        self.params = {"means": np.zeros([train_data[0].shape[0], 2]),
                       "stds": np.zeros([train_data[0].shape[0], 2]),
                       "examples": np.zeros([1, 2]),
                       "smooth":smooth
                       }
        
        #set parameters according to initial data
        for c in range(2):
            c_data = [data for data_index, data in enumerate(train_data) if train_target[data_index] == c]
            if len(c_data) != 0 :
                self.params["means"][:, c] =  np.sum(c_data, axis=0)
                self.params["stds"][:, c] = len(c_data) * np.var(c_data, axis=0)
                self.params["examples"][:, c] = len(c_data)
        
    def partial_fit(self, train_data, train_target):
        '''
        Update NB parameters according received train_data and train_target

        Args:
            train_data (numpy array of lists): feature vectors
            train_target (numpy array of ints): values 1 and 0, labels
        
        Returns:
            None
        '''

        for c in range(2):
            c_data = [data for data_index, data in enumerate(train_data) if train_target[data_index] == c]
            if len(c_data) != 0 and self.params["examples"][:, c] != 0:
                number_of_new_examples = len(c_data)
                self.params["stds"][:, c] = self.params["stds"][:, c] + len(c_data) * np.var(c_data, axis=0) + (self.params["examples"][:, c] / (number_of_new_examples * (number_of_new_examples + self.params["examples"][:, c]))) * ((number_of_new_examples / self.params["examples"][:, c]) * self.params["means"][:, c]-np.sum(c_data, axis=0))**2
                self.params["means"][:, c] =  self.params["means"][:, c] + np.sum(c_data, axis=0)
                self.params["examples"][:, c] = self.params["examples"][:, c] + len(c_data)
            # we received first example of label c
            if len(c_data) != 0 and self.params["examples"][:, c] == 0:
                self.params["means"][:, c] =  np.sum(c_data, axis=0)
                self.params["stds"][:, c] = len(c_data) * np.var(c_data, axis=0)
                self.params["examples"][:, c] = len(c_data)
        
    def predict(self, data, feature_suitability = 0):
        '''
        Predict label to data using exact part of the data feature vector specified in feature_suitability
        
        Args:
            data (numpy array of floats): single feature vector
            feature_suitability (list of bools): determining whether use feature with according index or not / 0 means do not deal with feature suitability=all features are suitable
        
        Returns:
            int, predicted label
        '''

        if 0 in self.params["examples"]:
            if self.params["examples"][:, 0] == 0:
                return 1
            else : 
                return 0
            
        means = self.params["means"].copy()
        stds = self.params["stds"].copy()

        #delete non suitable parts of the sample and corresponding parameters
        if feature_suitability != 0:
            for feature_index in range(len(data[0]) - 1, -1, -1):
                if not feature_suitability[feature_index]:
                    means = np.delete(means, feature_index, 0)
                    stds = np.delete(stds, feature_index, 0)
                    data = np.delete(data,feature_index)
        
            #whole data is not suitable
            if len(data) == 0:
                return random.randint(0,1)
        
            if False in feature_suitability: 
                data = np.array([data])
    
        #compute probabilities
        log_probabilities = np.zeros((len(data), 2))

        if self.params["smooth"]:
            log_probabilities += np.log(0.2*np.ones([1, 2]) + 0.6*self.params["examples"] / np.sum(self.params["examples"]) )
        else:
            log_probabilities += np.log(self.params["examples"] / np.sum(self.params["examples"]) )

        log_probabilities += np.sum(stats.norm(loc = (1 / self.params["examples"]) * means, scale =  np.sqrt((1 / self.params["examples"]) * stds + 0.001)).logpdf(np.expand_dims(data, -1)), axis=1)
        
        y_pred = np.argmax(log_probabilities, axis=1)[0]

        return y_pred


class DWM:
    '''
    Dynamic Weighted Majority algorithm using NB classifier as expert, with possible label delay and feature subset drift indicators
    '''

    def __init__(self,samples,labels,update_period, weight_factor, number_of_categories, weight_treshol, delay, learning_period, smooth=False, drift_detector = False) -> None:
        super().__init__()
        '''
        Class constructor, initialize parameters

        Args:
            samples (numpy array): consisting lists of floats standing for feature vectors for initialization of first expert
            labels (numpy array of ints), labels 0 or 1, according to samples
            update_period (int): period of ensemble and weights update
            weight_factor (float): between 0 and 1, factor for expert weight when wrong local prediction
            number_of_categories (int): number of distinct labels (expected number is 2)
            weighted_threshold (float): removal threshold for expert weight
            delay (int): label delay -we expect to get label at the time, but we use it with this delay
            learning_period (int): number of last labeled samples we provide the new expert for initial learning
            smooth (bool): whether apply label prior smoothing or not (when predict using Gaussian Naive Bayes classifier)
            drift_detector (bool): - False - do not use any drift detectors  
                                   - [1,parameters_for_hellinger_detection] - we will use class Hellinger_detection for feature suitability detection, parameters_for_hellinger_distace is list consisting of initialization parameters for this class        
                                   - [2,parameters_for_GMM_detection] - we will use class GMM_overal_process for feature suitability detection, parameters_for_GMM_detection is list consisting of initialization parameters for this class

        
        '''
    
        #determine which drift detector (if any) use
        type_detect = False
        detect_params = False
        if drift_detector != False:
            if drift_detector[0] == 1:
                type_detect = 1
                detect_params = drift_detector[1]
                drift_detector = Hellinger_detection(detect_params)
            elif drift_detector[0] == 2:
                type_detect = 2
                detect_params = drift_detector[1]
                drift_detector = GMM_overal_process(detect_params)

        self.DWM_params = {   'experts' : [NBModel(samples,labels,smooth)],    #list of structures with methods: predict(X) returning int 0,...,'number_of_categories'-1, for float filled numpy array feature vector X
                              #                                                                                        : partial_fit(X,y) increment parameters according to new feature vetor X and label y (0/1)
                              'weights' : [1],                          #list of floats between 0 and 1, determining weight of expert on according index in 'experts'
                              'local_pred_hist' : [[]],                 #list of lists of ints, where each sublists corresponds to local prediction of expert on same index in 'experts
                              'glob_pred_history' : [],                 #list of ints, last delay + 1 global predictions
                              'weight_factor' : weight_factor,
                              'sample_history': [],                     #list of seen samples whose labels are yet unknown together with samples needed for learning period
                              'label_history': [],                      #list of labels needed for learning period
                              'number_of_categories' : number_of_categories,
                              'drift_detector' : [drift_detector],      #class with method .process_sample(X), for float filled numpy array, returning list of bools determining whether use according feature for classification or not 
                              'weight_treshol' : weight_treshol,
                              'learning_period' : learning_period,
                              'processed_samples' : 1,
                              'delay' : delay,                          
                              'p' : update_period,                                  
                              'type_of_detector': type_detect,          
                              'detector_params': detect_params,
                              'smooth': smooth       
        }

        #increment drift detection algorithm if there is any
        if self.DWM_params["drift_detector"][0] != False:
            for data in samples:
                self.DWM_params["drift_detector"][0].process_sample(data,self)

    def add_to_local_predictions_history(self, local_pred, index):
        '''
        Keep necessary history of predictions in the situation of the label delay
        
        Args:
            local_pred (int): locally predicted label
            index (int): index of according expert
        
        Returns:
            None
        '''
    
         
        if len(self.DWM_params['local_pred_hist'][index]) > self.DWM_params['delay']:
            self.DWM_params['local_pred_hist'][index] = self.DWM_params['local_pred_hist'][index][1:] + [local_pred]
        else:
            self.DWM_params['local_pred_hist'][index].append(local_pred)

    def process_sample(self,single_data,delayed_label):
        '''
        Overal process of single sample through DWM algorithm

        Args:
        single_data (numpy array of floats): feature vector
        delayed_label (int): label according to feature vetcor processed 'delay' number of step in history, equals -1 if there is no label due to delay at the beginning

        Returns:
            int, predicted label of the processed feature vector
        '''
      
        #at the beginning when when delay > 1, we receive only samples without any label
        if delayed_label[0] == -1:
            self.DWM_params['sample_history'].append(single_data)

            #determine suitability according to drift detector if there is any
            if not self.DWM_params["drift_detector"][0]:
                feature_suitability = 0
            else:
                feature_suitability = self.DWM_params["drift_detector"][0].process_sample(single_data[0],self.DWM_params['experts'][0])

            return self.DWM_params['experts'][0].predict(single_data,feature_suitability)
        
        #process samples using ensemble of experts
        weighted_sum = [0] * self.DWM_params['number_of_categories']
        for expert_index, expert in enumerate(self.DWM_params['experts']):
            if not self.DWM_params["drift_detector"][0]:
                feature_suitability = 0
            else:
                feature_suitability = self.DWM_params["drift_detector"][expert_index].process_sample(single_data[0],expert)
            local_prediction = int( expert.predict(single_data,feature_suitability))
            #add local prediction of expert into history of local predictions so we can compare it with golden labels in future 
            DWM.add_to_local_predictions_history(self,local_prediction,expert_index)
            
            #if upadte period is met and expert collected enough predictions then update weight based on prediction correctnes
            if self.DWM_params['processed_samples'] % self.DWM_params['p'] == 0:
                if len(self.DWM_params['local_pred_hist'][expert_index]) == self.DWM_params['delay'] + 1 and self.DWM_params['local_pred_hist'][expert_index][0] != delayed_label[0]:
                    self.DWM_params['weights'][expert_index] = self.DWM_params['weight_factor'] * self.DWM_params['weights'][expert_index]
            
            weighted_sum[local_prediction] += self.DWM_params['weights'][expert_index]

        global_prediction = np.argmax( weighted_sum )

        #update memory of global prediction history, seen samples and labels, this step is necessary due to label delay
        if len(self.DWM_params['glob_pred_history']) == self.DWM_params['delay'] + 1:
            self.DWM_params['glob_pred_history'] = self.DWM_params['glob_pred_history'][1:] + [global_prediction]
        else:
            self.DWM_params['glob_pred_history'].append(global_prediction)
        
        if len(self.DWM_params['sample_history']) == self.DWM_params['delay'] + self.DWM_params['learning_period']:
            self.DWM_params['sample_history'] = self.DWM_params['sample_history'][1:] + [single_data[0]]
        else:
            self.DWM_params['sample_history'].append(single_data[0])
        
        if len(self.DWM_params['label_history']) == self.DWM_params['learning_period']:
            self.DWM_params['label_history'] = self.DWM_params['label_history'][1:] + [delayed_label[0]]
        else:
            self.DWM_params['label_history'].append(delayed_label[0])
        
        #update experts using latest received label and according feature vector from history
        for expert in self.DWM_params['experts']:
            if self.DWM_params['processed_samples'] > self.DWM_params['delay']:
                expert.partial_fit(np.array([self.DWM_params['sample_history'][- self.DWM_params['delay'] - 1]]),delayed_label)


        #if update period is met update weights and ensemble
        if self.DWM_params['processed_samples'] % self.DWM_params['p'] == 0:
            normalize_factor = np.max(self.DWM_params['weights'])
            for weight_index in range(len(self.DWM_params['weights'])):
                self.DWM_params['weights'][weight_index] = self.DWM_params['weights'][weight_index] / normalize_factor
            for weight_index in range(len(self.DWM_params['weights']) - 1, -1, -1):
                if self.DWM_params['weights'][weight_index] < self.DWM_params['weight_treshol']:
                    del self.DWM_params['weights'][weight_index]
                    del self.DWM_params['experts'][weight_index]
                    del self.DWM_params['local_pred_hist'][weight_index]
                    if self.DWM_params["drift_detector"][0] != False:
                        del self.DWM_params["drift_detector"][weight_index]
            if len(self.DWM_params['glob_pred_history']) == self.DWM_params['delay'] + 1 and self.DWM_params['glob_pred_history'][0] != delayed_label:
                self.DWM_params['experts'].append(NBModel(self.DWM_params['sample_history'][:len(self.DWM_params['label_history'])],self.DWM_params['label_history'],self.DWM_params['smooth']))
                self.DWM_params['weights'].append(1)
                self.DWM_params['local_pred_hist'].append([])
                if self.DWM_params["drift_detector"][0] != False:
                    if self.DWM_params["type_of_detector"] == 1:
                        self.DWM_params["drift_detector"].append(Hellinger_detection(self.DWM_params["detector_params"]))
                    else:
                        self.DWM_params["drift_detector"].append(GMM_overal_process(self.DWM_params["detector_params"]))

        self.DWM_params['processed_samples'] += 1

        return global_prediction

    def get_number_of_models(self):
        '''
        Returns number of experts in ensemble DWM algortihm uses at the moment

        Returns:
            int, number of the experts
        '''
        return len(self.DWM_params['experts'])
    
    def get_predictions_of(self,samples):
        '''
        Return list of predictions for given samples using current experts and weights of the DWM

        Args:
            samples (numpy array of lists of floats): feature vectors

        Return:
            list of ints, label predictions of the given feature vectros
        '''

        predictions = []
        for sample_index in range(len(samples)):
            weighted_sum = [0] * self.DWM_params['number_of_categories']
            for expert_index, expert in enumerate(self.DWM_params['experts']):
                local_prediction = int( expert.predict(samples[sample_index : sample_index + 1]))
                weighted_sum[local_prediction] += self.DWM_params['weights'][expert_index]
            predictions.append(np.argmax( weighted_sum ))
        
        return predictions

    def get_accuracy_on_samples(self,test_data, test_target):
        '''
        Return percentage accuracy of predictions using current DWM params on test_data and test_target

        Args:
            test_data (numpy array of lists of floats): feature vectors
            test_target (list of ints): labels corresponding to given feature vectors
        
        Returns:
            float, accuracy on the given data
        '''

        predictions = DWM.get_predictions_of(self,test_data)
        correct = 0
        for prediction_index in range(len(predictions)):
            if predictions[prediction_index] == test_target[prediction_index]:
                correct += 1
        return correct / len(predictions)


class Process_single_feature_subset:
    '''
    Hellinger proces for single feature subset
    '''
    def __init__(self, gamma, window_size, size_params) -> None:
        '''
        Class constructor
        
        Args:
            gamma (float): gamma parameter for Hellinger concept drift detection
            window_size (int): window size
            size_params (list of ints): - [0], minimal number measured differences, before drift detection
                                        - [1], minimal number processed by histogram
                                        - [2], maximal number processed by histogram

        '''

        super().__init__()
        self.Hellinger_gamma = gamma
        self.Hellinger_window_size = window_size
        self.Hellinger_distribution_l = []
        self.Hellinger_epsilons = []
        self.Hellinger_new_window = []
        self.drifted_feature = 0

        self.sum_of_differences = 0
        self.sum_for_variance = 0
        self.number_of_processed_differences = 0
        self.epsilon = 0

        self.minimal_estimation_size = size_params[0]
        self.max_histogram = size_params[2]
        self.min_histogram = size_params[1]
        self.num_bins_histogram = int(np.floor(np.sqrt(self.Hellinger_window_size))) #number of bins when no prior information usual set  for each window to sqrt of the window size

    def generate_basic_histogram(self,window_feature_subset):
        '''
        Generate basic historgram for feature subset of samples in self.Hellinger_new_window 

        Args:
            window_feature_subset (numpy array of lists of floats): restricted feature vectors
        
        Returns:
            list of lists of ints, historgrams for feature vectors given by number of instances in each bin

        
        '''
        distribution_bins = []
        num_bin = self.num_bins_histogram 
        for feature in range(len(window_feature_subset[0])):
            single_feature = [window_feature_subset[j][feature] for j in range(len(window_feature_subset))]
            max = self.max_histogram
            min = self.min_histogram 
            single_feature.sort()
            step = np.abs((max - min) / num_bin)
            bin = [0] * num_bin
            i = 0
            j = 0
            if single_feature[0] < min: #process outliers lower then expected minimum
                while single_feature[j] < min and j < self.Hellinger_window_size:
                    j += 1
                    bin[i] += 1

            while j < self.Hellinger_window_size:
                if single_feature[j] >= min + (i + 1) * step and i < num_bin - 1:
                    i += 1
                if single_feature[j] < min + (i + 1) * step:
                    bin[i] += 1
                    j += 1

                elif i == num_bin - 1 and single_feature[j] >= max: #process outliers which are greater then expected max
                    bin[i] += 1
                    j += 1
                
            distribution_bins.append(bin)
        return distribution_bins

    def hellinger_distance(self, P, Q): 
        '''
        Returns Hellinger distance

        Args:
            P (list of lists of ints): histogram
            Q (list of lists of ints): histogram
        
        Returns:
            float, Hellinger distance of P and Q
        '''
        #suppose same number of bins in P and Q
        features_distance = []
        for i in range(len(P)):
            sum_P = np.sum(P[i])
            sum_Q = np.sum(Q[i])
            feature_sum = 0
            for j in range(len(P[i])):
                feature_sum += (np.sqrt(P[i][j]/sum_P) - np.sqrt(Q[i][j]/sum_Q))**2
            features_distance.append(feature_sum)
        return (1 / len(P)) * np.sum([np.sqrt(features_distance[feature]) for feature in range(len(P))])
    
    def add_value_to_distribution(self,new_value,feature):
        '''
        Add single new value to existing histogram

        Args:
            new_value (float): new value of the feature
            feature (int): index of the feature
        
        Returns:
            None
        '''

        max = self.max_histogram
        min = self.min_histogram 
        num_of_bins = self.num_bins_histogram 
        bin_value = int(np.floor((new_value - min) / ((max - min) / num_of_bins)))
        if bin_value >= num_of_bins:
            bin_value = num_of_bins - 1
        if bin_value < 0:
            bin_value = 0
        self.Hellinger_distribution_l[feature][bin_value] += 1

    def reset_process(self):
        '''
        Reset all values describing current concept (we detected concept drift)
        
        Returns:
            None
        '''

        self.Hellinger_new_window = []
        self.Hellinger_distribution_l = []
        self.sum_of_differences = 0
        self.sum_for_variance = 0
        self.number_of_processed_differences = 0
    
    def get_epsilon(self):
        '''
        Return current epsilon parameter

        Returns:
            float, epsilon
        '''

        return self.epsilon

    def proces_data_single_feature_subset(self,data):
        '''
        Main method of the class, processses single restricted data (to examined feature subset) through Hellinger process as subprocess of main overall class Hellinger_detection

        Args:
            data (numpy array of floats): restricted feature vectors
        
        Returns:
            bool or 0, indicator whether was concept drift detected in the examined feature subset (True), or not (False), if we do not process enought data to decide it returns 0
        '''

        #Hellinger distance concept drift detection
        #if collected enought differences, initialize values for estimation
        if len(self.Hellinger_epsilons) == self.minimal_estimation_size:
            self.sum_of_differences = np.sum(self.Hellinger_epsilons, axis=0)
            self.sum_for_variance = len(self.Hellinger_epsilons) * np.var(self.Hellinger_epsilons, axis=0)
            self.number_of_processed_differences = len(self.Hellinger_epsilons)
            self.Hellinger_epsilons = []

        self.Hellinger_new_window.append(data)
        if self.Hellinger_distribution_l == []: #we are at the very beginning of the process or after drift
            if len(self.Hellinger_new_window) == self.Hellinger_window_size:
                self.Hellinger_distribution_l = Process_single_feature_subset.generate_basic_histogram(self,self.Hellinger_new_window)
                self.Hellinger_new_window = []
            self.drifted_feature = 0

        elif len(self.Hellinger_new_window) == self.Hellinger_window_size  and self.number_of_processed_differences >= self.minimal_estimation_size:
            new_distribution = Process_single_feature_subset.generate_basic_histogram(self,self.Hellinger_new_window)
            delta = Process_single_feature_subset.hellinger_distance(self,new_distribution,self.Hellinger_distribution_l)
            self.epsilon = np.abs(delta)

            eps_hat = (1 / self.number_of_processed_differences) * self.sum_of_differences
            sig_hat = np.sqrt((1 / self.number_of_processed_differences ) * self.sum_for_variance)

            if np.abs(self.epsilon) > eps_hat + self.Hellinger_gamma * sig_hat:
                self.drifted_feature = True
                Process_single_feature_subset.reset_process(self)
            else:
                #drift not detected actualize parameters accordingly
                for feature in range(len(self.Hellinger_new_window[0])):
                    #extend histogram by oldest value of the window
                    new_value = self.Hellinger_new_window[0][feature]
                    Process_single_feature_subset.add_value_to_distribution(self,new_value,feature)
                self.drifted_feature = False
                self.Hellinger_new_window =  self.Hellinger_new_window[1:]
                
                self.sum_for_variance = self.sum_for_variance +  (self.number_of_processed_differences / (1 + self.number_of_processed_differences)) * ((1 / self.number_of_processed_differences) * self.sum_of_differences - self.epsilon)**2
                self.number_of_processed_differences = self.number_of_processed_differences + 1
                self.sum_of_differences = self.sum_of_differences + self.epsilon

        elif len(self.Hellinger_new_window) == self.Hellinger_window_size  and len(self.Hellinger_epsilons) < self.minimal_estimation_size and self.number_of_processed_differences == 0:
            new_distribution = Process_single_feature_subset.generate_basic_histogram(self,self.Hellinger_new_window)
            delta = Process_single_feature_subset.hellinger_distance(self,new_distribution,self.Hellinger_distribution_l)
            self.epsilon = np.abs(delta)
            self.Hellinger_epsilons.append(self.epsilon)
            for feature in range(len(self.Hellinger_new_window[0])):
                new_value = self.Hellinger_new_window[0][feature]
                Process_single_feature_subset.add_value_to_distribution(self,new_value,feature)
            self.drifted_feature = 0
            self.Hellinger_new_window =  self.Hellinger_new_window[1:]      
        else:
            self.drifted_feature = 0

        return self.drifted_feature   
         

class Hellinger_detection:
    '''
    Algorithm 3
    Main class for Hellinger concept drift detection in feature subsets
    '''
    def __init__(self, Hellinger_distance_params) -> None:
        '''
        Constructor of the class, initialize parameters
        
        Args:
            Hellinger_distance_params (list): consisting of
                 - Hellinger_distance_params[0], int, window size
                 - Hellinger_distance_params[1], list of lists of ints, list giving sets of dependent features
                 - Hellinger_distance_params[2], float, gamma parameter for Hellinger concept drift detection
                 - Hellinger_distance_params[3], int, number of steps when the detected drift persists (number of steps after drift detection, we consider subset to be non-suitable for detection)
                 - Hellinger_distance_params[4], list of ints  [0] - minimal number measured differences, before drift detection
                                                               [1] - minimal number for histogram (numbers under this number goes automaticaly to 1 bin)
                                                               [2] - maximal number for histogram (numbers above this number goes automaticaly to last bin)
        '''
        super().__init__()

        #initialize parameters
        self.Hellinger_featur_subsets = Hellinger_distance_params[1]
        self.Hellinger_featur_subsets_processes = []    #list consisting subprocesses for exact given feature subsets
        self.drifted_feature = [0 for _ in range(len(self.Hellinger_featur_subsets))]   #list keeping information about drift, 0 = we have no information,
        #                                                                                                                      True = according feature subset is drifted
        #                                                                                                                      False = we consider feature subset as stable
        self.drift_persisting = [0 for _ in range(len(self.Hellinger_featur_subsets))]  #determine how long we would consider subset as drifted
        self.persisting_period = Hellinger_distance_params[3]
        for _ in range(len(self.Hellinger_featur_subsets)):
            self.Hellinger_featur_subsets_processes.append(Process_single_feature_subset(Hellinger_distance_params[2],Hellinger_distance_params[0],Hellinger_distance_params[4]))

    def get_epsilons(self):
        '''
        Return current epsilon parameter of all feature subsets

        Returns:
            list of float, epsilons of the feature subsets
        '''

        epsilons = []
        for feature_subset in range(len(self.Hellinger_featur_subsets)):
            epsilons.append([feature_subset,self.Hellinger_featur_subsets_processes[feature_subset].get_epsilon()])
        return epsilons

    def get_last_subset_drift(self):
        '''
        Return information about drift in each feature subset from the last step of the process

        Returns:
            list of bools and 0, indicators wheteher drift was detected in last step (was = True, was not = False, not enought information = 0)
        '''

        return self.drifted_feature
    
    def process_sample(self, data, classifier = False):
        '''
        Divide feature vector into predefined feature subset and then process it througt Hellinger concept drift detection using class Process_single_feature_subset

        Args:
            data (numpy array of floats): single feature vectors
            classifier (any): not used due to compatibility and possible extension
        
        Returns:
            list of bools and 0: indicators wheteher drift was detected processing input data  (was = True, was not = False, not enought information = 0)

        '''

        #divide data into feature subsets
        data_divided_into_feature_subsets = []
        for feature_subset in range(len(self.Hellinger_featur_subsets)):
            data_divided_into_feature_subsets.append([data[feature] for feature in self.Hellinger_featur_subsets[feature_subset]])

        #proces one step in each of the hellinger processes and get information about drift
        for feature_subset in range(len(self.Hellinger_featur_subsets)):
            self.drifted_feature[feature_subset] = self.Hellinger_featur_subsets_processes[feature_subset].proces_data_single_feature_subset(data_divided_into_feature_subsets[feature_subset])
        
        #determine feature suitability for each feature, we assume that only drifted features are not suitable
        feature_suitability = [True] *  np.sum([len(subset) for subset in self.Hellinger_featur_subsets])
        for subset_index, subset in enumerate(self.Hellinger_featur_subsets):
            if self.drifted_feature[subset_index]:
                self.drift_persisting[subset_index] = self.persisting_period
            if self.drift_persisting[subset_index] > 0:
                self.drift_persisting[subset_index] -= 1
                for feature in subset:
                    feature_suitability[feature] = False
        
        return feature_suitability
    

def main() -> float:
    return 0

if __name__ == "__main__":
    main()
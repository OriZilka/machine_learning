import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_value_dataset = []

        for i in range (self.dataset.shape[0]):
            if self.dataset[i, -1] == self.class_value:
                self.class_value_dataset.append(self.dataset[i, :])
        
        self.class_value_dataset= np.array(self.class_value_dataset)    

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        dataset_size = self.dataset.shape[0]
        class_value_dataset_size = (self.class_value_dataset).shape[0]
        
        return (class_value_dataset_size/ dataset_size)
    

    def mean_stdd(self):

        class_value_dataset_size = self.class_value_dataset.shape[0]
        temp_data = self.class_value_dataset[:,0]
        humid_data = self.class_value_dataset[:,1]

        temp_sum = np.sum(temp_data)
        humid_sum = np.sum(humid_data)

        temp_mean = temp_sum / class_value_dataset_size
        humid_mean = humid_sum / class_value_dataset_size

        temp_data_minus_mean = temp_data - temp_mean
        humid_data_minus_mean = humid_data - humid_mean

        sum_temp_data_minus_mean = np.sum(temp_data_minus_mean ** 2)
        sum_humid_data_minus_mean = np.sum(humid_data_minus_mean ** 2)

        temp_stdd = np.sqrt(sum_temp_data_minus_mean / class_value_dataset_size)
        humid_stdd = np.sqrt(sum_humid_data_minus_mean / class_value_dataset_size)

        return temp_mean, humid_mean, temp_stdd, humid_stdd

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood porbability of the instance under the class according to the dataset distribution.
        """
        temp_mean, humid_mean, temp_stdd, humid_stdd = self.mean_stdd()
        temp_likelihood = normal_pdf(x[0], temp_mean, temp_stdd)
        humid_likelihood = normal_pdf(x[1], humid_mean, humid_stdd)
      
        return (temp_likelihood * humid_likelihood)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()

        return (likelihood * prior)

    
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_value_dataset = []

        for i in range (self.dataset.shape[0]):
            if self.dataset[i, -1] == self.class_value:
                self.class_value_dataset.append(self.dataset[i, :])
        
        self.class_value_dataset= np.array(self.class_value_dataset)    

        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        dataset_size = self.dataset.shape[0]
        class_value_dataset_size = (self.class_value_dataset).shape[0]
        
        return (class_value_dataset_size/ dataset_size)
    
    def mean_cov(self):
        
        class_value_dataset_size = self.class_value_dataset.shape[0]
        temp_data = self.class_value_dataset[:,0]
        humid_data = self.class_value_dataset[:,1]

        temp_sum = np.sum(temp_data)
        humid_sum = np.sum(humid_data)

        temp_mean = temp_sum / class_value_dataset_size
        humid_mean = humid_sum / class_value_dataset_size

        cov = np.cov(self.class_value_dataset[:,[0,1]].T)

        return (np.array([temp_mean, humid_mean]), cov)

    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        mean, cov = self.mean_cov()
        likelihood = multi_normal_pdf(x, mean, cov)

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return (self.get_instance_likelihood(x) * self.get_prior())
    
   

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    squareStdd = std ** 2
    square_x_minus_mean = np.square(x-mean)
    numerator = np.power(np.e, -(square_x_minus_mean / (2 * squareStdd)))
    dinomerator = np.sqrt(2 * (np.pi) * squareStdd)

    return (numerator / dinomerator)

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    # part1 components
    power_two_pi_minusOne = np.power(2 * np.pi, -1)
    power_detCov_minusHalf = np.power(np.linalg.det(cov),-1/2)

    x = x[:-1]

    # part2 components
    x_minus_mean = (x - mean)
    cov_power_minusOne = np.linalg.inv(cov)
    full_part2 = np.power(np.e, -(0.5 * np.matmul(np.matmul(x_minus_mean.T, cov_power_minusOne), x_minus_mean)))

    res = (power_two_pi_minusOne) * (power_detCov_minusHalf) * (full_part2)

    return res


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_value_dataset = []

        for i in range (self.dataset.shape[0]):
            if self.dataset[i, -1] == self.class_value:
                self.class_value_dataset.append(self.dataset[i, :])
        
        self.class_value_dataset= np.array(self.class_value_dataset)   
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        dataset_size = self.dataset.shape[0]
        class_value_dataset_size = (self.class_value_dataset).shape[0]
        
        return (class_value_dataset_size/ dataset_size)
    
    def calcVj(self, col):

        return np.unique(self.class_value_dataset[:,col], return_counts=True) 
    
    def calc_nij(self, xj, col):

        return list(self.class_value_dataset[:, col]).count(xj)

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        x = x[:-1]
        n_i = self.class_value_dataset.shape[0]
        
        for col in range(x.shape[0]):
            v_j , none = self.calcVj(col)
            size_v_j = len(v_j)
            n_ij = self.calc_nij(x[col], col)
            P_xj_Ai = (n_ij + 1) / (n_i + size_v_j)
            likelihood *= P_xj_Ai
        
        return likelihood
    

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """

        current_posterior = self.get_instance_likelihood(x) * self.get_prior()

        if (self.class_value == 0):
            other_class = DiscreteNBClassDistribution(self.dataset, 1)
        else:
            other_class = DiscreteNBClassDistribution(self.dataset, 0)
        
        other_posterior = other_class.get_instance_likelihood(x) * other_class.get_prior()
        full_posterior = other_posterior + current_posterior
        resalut = current_posterior / full_posterior

        return resalut

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1


    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """

        ccd0_postrior = self.ccd0.get_instance_posterior(x)
        ccd1_postrior = self.ccd1.get_instance_posterior(x)
        
        if(ccd0_postrior > ccd1_postrior):
            return 0
        else:
            return 1
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    testset_size = testset.shape[0]
    correctly_classified = 0

    for i in range(testset_size):
        pridict_i = map_classifier.predict(testset[i,:])
        if pridict_i == testset[i,-1]:
            correctly_classified = correctly_classified + 1
    
    return (correctly_classified / testset_size)
    
            
            
            
            
            
            
            
            
            
    
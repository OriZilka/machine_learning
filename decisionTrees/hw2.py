import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    #size = S, numOfOnes = s1, numOfZeros = s2
    if (len(data) < 2):
        return 0
    size = data.shape[0]
    numOfOnes = np.count_nonzero(data[:, -1])
    numOfZeros = size - numOfOnes
    gini = 1 - (np.square(numOfOnes/size) + np.square(numOfZeros/size))
    
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    #size = S, numOfOnes = s1, numOfZeros = s2
    if (len(data) < 2):
        return 0
    size = len(data[:,-1])
    numOfOnes = np.count_nonzero(data[:,-1])
    numOfZeros = size - np.count_nonzero(data[:,-1])
    
    if (numOfOnes/size == 0 or numOfZeros/size == 0):
        return 0
    
    entropy = - ( (numOfOnes/size)*np.log2(numOfOnes/size) +
                  (numOfZeros/size)*np.log2(numOfZeros/size) ) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.prediction = None
        self.left_child = None 
        self.right_child = None
        self.count = None # counts how many instances of the prediction value.
        self.isLeaf = False
    
def chi_square_Boolean(fData, lCData, rCData, chi_value):
    
    if chi_value==1 : return True
    
    size_of_data = fData.shape[0]
    size_of_left_data = lCData.shape[0]
    size_of_right_data = rCData.shape[0]
    
    p = np.count_nonzero(fData[:,-1])
    n = (size_of_data) - p
    
    p0 = np.count_nonzero(lCData[:,-1])
    p1 = np.count_nonzero(rCData[:,-1])
    
    n0 = (size_of_left_data) - p0
    n1 = (size_of_right_data) - p1
    
    E00 = size_of_left_data * (p / size_of_data)
    E10 = size_of_left_data * (n / size_of_data)
    E01 = size_of_right_data * (p / size_of_data)
    E11 = size_of_right_data * (n / size_of_data)
    
    return ((  ((np.square(p0 - E00)) / E00) + (np.square(n0 - E10) / E10)  + (np.square(p1 - E01) / E01) + (np.square(n1 - E11) / E11) ) > (chi_table[chi_value]))
    
    
    
def build_tree(data, impurity, chi_value = 1):          
            
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    best_feature, best_threshold, best_goodness, data_of_left_child, data_of_right_child = best_split(data, impurity)
    root = DecisionNode(best_feature, best_threshold)
    build_tree_rec(data, root, impurity, chi_value)
    root.prediction = predict2(root, data[:, -1])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def build_tree_rec(data, node, impurity, chi_value):  
    
    #checking if the node received as an argument is a leaf - step 1
    if(impurity(data) == 0):
        node.prediction = predict2(node, data[:, -1])
        node.isLeaf = True
        return
    
    #activating best_split on the data received as an argument - step 2
    best_feature, best_threshold, best_goodness, data_of_left_child, data_of_right_child = best_split(data, impurity)
    
    # step 3- checking if the chi_value is larger than threshold.
    if(not chi_square_Boolean(data, data_of_left_child, data_of_right_child, chi_value)):
        node.prediction = predict2(node, data[:, -1]) 
        node.isLeaf = True
        return
    
    #bulidng left node - step 4.1
    best_feature_of_left_node, best_threshold_of_left_node, best_goodness_of_left_node, data_of_node_of_left_child, data_of_node_of_right_child = best_split(data_of_left_child, impurity)
    left_child = DecisionNode(best_feature_of_left_node, best_threshold_of_left_node)
    

    node.left_child = left_child
    left_child.prediction = predict2(left_child, data[:, -1])
 

    #building right node - step 4.2
    best_feature_of_right_node, best_threshold_of_right_node, best_goodness_of_right_node, data_of_node_of_left_child, data_of_node_of_right_child = best_split(data_of_right_child, impurity)
    right_child = DecisionNode(best_feature_of_right_node, best_threshold_of_right_node)
    node.right_child = right_child
    right_child.prediction = predict2(right_child, data[:, -1])

    #activating build_tree_rec recursively on the left and right children - step 5
    build_tree_rec(data_of_left_child, left_child, impurity, chi_value)
    build_tree_rec(data_of_right_child, right_child, impurity, chi_value)

        
        
# finds the best feature and threshold 
def best_split(data, impurity):
    
    best_feature = None
    best_goodness = 0
    best_threshold = None
    best_data_of_left_child = []
    best_data_of_right_child = []
    
    for feature in range((data.shape[1]) - 1):
        threshold, goodness, data_of_left_child, data_of_right_child = find_best_threshold(data, feature, impurity)
        if (goodness > best_goodness):
            best_feature = feature
            best_threshold = threshold
            best_goodness = goodness
            best_data_of_left_child = data_of_left_child
            best_data_of_right_child = data_of_right_child
    
    return best_feature, best_threshold, best_goodness, np.array(best_data_of_left_child), np.array(best_data_of_right_child)



# find best threshold from a feature.
def find_best_threshold(data, feature, impurity):
    threshold = []
    
    #sort the column of the current feature
    feature_data = np.sort(data[:, feature])
    
    #calc the average of each consecutive pair of values
    for i in range(len(feature_data) - 1):
        threshold.append((feature_data[i] + feature_data[i+1]) / 2.0)

    max_goodness = 0
    max_threshold = 0
    data_of_left_child = []
    data_of_right_child = []
    
    #running over threshold list in order to find the best threshold     
    for i in range(len(threshold)):
        current_threshold = threshold[i]
        data_smaller_than_threshold , data_higher_than_threshold = split_data(data, feature, current_threshold)
        size_of_data_smaller_than_threshold = data_smaller_than_threshold.shape[0]
        size_of_data_higher_than_threshold = data_higher_than_threshold.shape[0]
        size_of_data = data.shape[0]
        
        #if right child: size_of_data_higher_than_threshold = |Sv|, size_of_data = |S|
        #if left child: size_of_data_smaller_than_threshold = |Sv|
        goodness_of_split_by_value = impurity(data) - ((size_of_data_higher_than_threshold/size_of_data)*impurity(data_higher_than_threshold)) - ((size_of_data_smaller_than_threshold/size_of_data)*impurity(data_smaller_than_threshold))
        
        if (goodness_of_split_by_value > max_goodness):
            max_goodness = goodness_of_split_by_value
            max_threshold = current_threshold
            data_of_left_child = data_smaller_than_threshold
            data_of_right_child = data_higher_than_threshold
            
    return max_threshold, max_goodness, data_of_left_child, data_of_right_child


#split the data by the current feature and threshold
def split_data(data, feature, threshold):
    data_smaller_than_threshold = []
    data_higher_than_threshold = []
    for i in range(data.shape[0]):
        if (data[i, feature] < threshold):
            data_smaller_than_threshold.append(data[i, :])
        else:
            data_higher_than_threshold.append(data[i, :])
    
    return np.array(data_smaller_than_threshold), np.array(data_higher_than_threshold)


# returns the max instances of predicted options.
def predict2(node, data):
        
    predict_options, counts = np.unique(data, return_counts=True)
    d = dict(zip(predict_options, counts))
    maximum = max(d,key=d.get)
    node.count = d.get(maximum)
    return maximum



def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    currNode = node
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    while(currNode.right_child and currNode.left_child):
        
        if (instance[currNode.feature] > currNode.value):
            currNode = currNode.right_child
        else: currNode = currNode.left_child
            
    pred = currNode.prediction
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    numOfCorrectPred = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    numOfInstances = dataset.shape[0]
    
    for instance in range(numOfInstances):
        if ((dataset[instance, -1]) == (predict(node, dataset[instance, :]))):
            numOfCorrectPred = numOfCorrectPred + 1
    
    accuracy = (numOfCorrectPred / numOfInstances)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return (accuracy*100)


def post_pruning(root, data):
    
    tree_acc_array = [calc_accuracy(root, data)]
    internal_nodes_array = [num_internal_nodes(root)]
    best_acc = 0
    best_parent = None
    numOfInternalNodesInTree = num_internal_nodes(root)

    
    while(root.left_child or root.right_child):
        
        best_acc = -1
        best_parent = None
        
        # an array with all the nodes that have 2 children that are leafs.
        possibleLeafParentsArray = possibleLeafParents(root)

        for parent in possibleLeafParentsArray:

            # saving the parents children
            temp_left_child = parent.left_child
            temp_right_child = parent.right_child

            # deleting the parents children from the tree
            parent.left_child = None
            parent.right_child = None

            # calculating the tree accuracy with out the children            
            curr_acc = calc_accuracy(root, data)

            # checking if the current split that I deleted was the best one to delete up until now.
            if (curr_acc > best_acc):
                best_acc = curr_acc
                best_parent = parent

            # returning the childrens value.
            parent.left_child = temp_left_child
            parent.right_child = temp_right_child

            numOfInternalNodesInTree = num_internal_nodes(root)

        
        # deletes the nodes that provide the best accuracy and turns their parent to a leaf.
        best_parent.left_child = None
        best_parent.right_child = None
        best_parent.isLeaf = True
        
        # updating the arrays that need to be returned
        internal_nodes_array.append(numOfInternalNodesInTree)
        tree_acc_array.append(best_acc)
    
    return (internal_nodes_array, tree_acc_array)

# counting all nodes in the tree except the leafs&the root
def num_internal_nodes(node):
    
    if (node.isLeaf):
        
        return 0
        
    return 1 + num_internal_nodes(node.left_child) + num_internal_nodes(node.right_child)



def possibleLeafParents(node):
    
    array = []
    queue = [node]
    while(len(queue)>0):
        curr_node = queue.pop(0)
        if (curr_node.right_child.isLeaf and curr_node.left_child.isLeaf):
            array.append(curr_node)
        if (not curr_node.left_child.isLeaf):
            queue.append(curr_node.left_child)
        if (not curr_node.right_child.isLeaf):
            queue.append(curr_node.right_child)
    
    return array
    
def print_tree(node, space = 0):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    
    s = "  " * space 
    if (node.isLeaf):
        print(s, end="")
        print("leaf: [{" + str(node.prediction) + ": " + str(node.count) + "}]")

    else:
        print(s, end="")
        print("[X" + str(node.feature) + " <= " + str(node.value) + "],")
        space += 1
        print_tree(node.left_child, space)
        if (node.right_child and node.left_child):
            print_tree(node.right_child, space)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

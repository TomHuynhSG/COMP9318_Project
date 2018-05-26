#Name: Tom & James
#COMP9318

import pandas
import helper


def get_freq_of_tokens(ls):
    tokens = {}
    for token in ls:
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens

def fool_classifier(test_data): ## Please do not change the function defination...
    #Training SVM (for testing purpose) using the class given
    strategy_instance = helper.strategy()

    #-----TURN DATA TO BAG OF WORDS-----

    #finding features of all paragraphs (class0 and class1)
    features = set()
    for i in strategy_instance.class0:
        features = features | set(i)

    for i in strategy_instance.class1:
        features = features | set(i)

    features = list(features) #changed features data type into a list 

    #creating a list of dict (every dict represent a paragraph)
    newlist0 = [] #newlist0 represent list of dict for class0 paragraph
    for i in strategy_instance.class0:
        newlist0.append(get_freq_of_tokens(i))

    newlist1 = [] #newlist1 represent list of dict for class1 paragraph
    for i in strategy_instance.class1:
        newlist1.append(get_freq_of_tokens(i))

    newdict = dict.fromkeys(features, 0) # make a dict with features as it's key with starting value is 0 for all keys

    #creating a list of dict combining all paragraph from class0 and class1 using a dict with all features
    #basically this is the Frequency Bag of Words    
    xdata = []
    for row in newlist0:
        tmp_dict = dict(newdict)
        for i in row:
            if i in newdict:
                tmp_dict[i]+= row[i]
        xdata.append(tmp_dict)

    for row in newlist1:
        tmp_dict = dict(newdict)
        for i in row:
            if i in newdict:
                tmp_dict[i]+= row[i]
        xdata.append(tmp_dict)

    #creating a list of y value
    ydata = []
    for i in range(len(newlist0)):
        ydata.append(0)
    for i in range(len(newlist1)):
        ydata.append(1)

    x_data = pandas.DataFrame(xdata) #changed the type of xdata into a pandas.dataframe
   
    #training SVM using default parameter
    parameter = {'gamma': 'auto', 'C': 0.005 ,'kernel': 'linear','degree': 3 ,'coef0': 0.0}
    
    #using own bag of words
    model = strategy_instance.train_svm(parameter, x_data, ydata)

    weights = model.coef_.tolist()

    #sorting and indexing the coefficients
    sortweights = list(reversed(sorted(weights[0])))
    idx_top = list()
    for i in range(len(sortweights)):
        idx = weights[0].index(sortweights[i])
        idx_top.append(idx)  
    
    #open the test data
    #test_data = "./test_data.txt"

    with open(test_data) as tdata:

        #list of paragraph
        list_par = list()
        for i in tdata:
            w = i.split()
            list_par.append(w)

         #check and modify every paragraph
        constant = 2 #nb of words will be inserted
        columns = list(x_data.columns.values) #list of features
        for i in range(len(list_par)):

            count_changes = 0 #count the nb of token changed

            #modify the words from the top features until count_changes == 20
            for w_ind in idx_top:

                if columns[w_ind] in list_par[i]:
                    #if positive words found
                    if weights[0][w_ind] > 0:
                        list_par[i] = list(filter(lambda a: a != columns[w_ind], list_par[i])) #removing features with (+) weight
                        count_changes += 1

                    #if negative words found
                    elif weights[0][w_ind] < 0: 
                        list_par[i] = list_par[i] + [columns[w_ind]]*constant #add constant number of negative words into it

                else:

                    #if negative word isn't there then add it
                    if weights[0][w_ind] < 0: 
                        list_par[i] = list_par[i] + [columns[w_ind]]*constant
                        count_changes += 1

                if count_changes == 20:
                    break
                    
    #create modified test text
    modified_data = "./modified_data.txt"
    with open(modified_data, "w+") as moddata:

        #change list_par into list of string
        list_string = [' '.join(x) for x in list_par]

        for i in range(len(list_string)):
            moddata.write(list_string[i]+"\n")
        

    
        
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    #modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


# Testing section
# test_data='./test_data.txt'
# strategy_instance = fool_classifier(test_data) 
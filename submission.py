#Name: Tom & James
#COMP9318

import helper
import pandas

def get_freq_of_tokens(ls):
    tokens = {}
    for token in ls:
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens

def fool_classifier(test_data): ## Please do not change the function defination...

    #Training SVM (for testing purpose)
    training = helper.strategy()

    #data provided in strategy class

    #turn data to bag of words (refer to wikipedia or other sources)

    #make each distinct word into a set
    features = set()
    for i in training.class0:
        features = features | set(i)

    for i in training.class1:
        features = features | set(i)

    features = list(features) #changed so the features is in order and can be used for other purpose

    #creating a dict per list (per doc)
    newlist0 = []
    for i in training.class0:
        newlist0.append(get_freq_of_tokens(i))

    newlist1 = []
    for i in training.class1:
        newlist1.append(get_freq_of_tokens(i))

    newdict = dict.fromkeys(features, 0)

    #make a list of dict representing the whole data
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

    ydata = []
    for i in range(len(newlist0)):
        ydata.append(0)
    for i in range(len(newlist1)):
        ydata.append(1)

    x_data = pandas.DataFrame(xdata)
    #x_train, and y_train done
    #or create a df with index (if you know exactly the number of rows), the use df.loc[x] to input the row
   
    #training SVM using default parameter
    parameter = {'gamma': 'auto', 'C': 1.0 ,'kernel': 'linear','degree': 3 ,'coef0': 0.0}
    
    #using own bag of words
    model = training.train_svm(parameter, x_data, ydata)

    weights = model.coef_.tolist()

    #apply abs to all of weights
    newweights = [abs(x) for x in weights[0]]

    #finding top 20
    sortweights = list(reversed(sorted(newweights)))
    top = sortweights[0:20]
    idx_top = list()
    for i in range(20):
        idx = newweights.index(top[i])
        idx_top.append(idx)  

    #finding out top 20 features words
    top_words = list()
    for i in range(20):
        top_words.append(features[idx_top[i]])

    #pair up words with their real coefficient
    top_words_coef = list()
    for i in range(20):
        top_words_coef.append((top_words[i],weights[0][idx_top[i]]))

    top_dict = dict(top_words_coef)
    
    #open the test data
    test_data = "D:/MyStudy/Data Warehousing & Data Mining/Project for Data Warehousing/test_data.txt"

    with open(test_data) as tdata:

        #list of paragraph
        list_par = list()
        for i in tdata:
            w = i.split()
            list_par.append(w)

        #check and modify every paragraph
        constant = 100 #nb of words will be inserted
        for i in range(len(list_par)):

            #check it against top 20 words
            for w in top_words:

                if w in list_par[i]:

                    #if positive words found
                    if top_dict[w] < 0: #originally ">"
                        list_par[i] = list(filter(lambda a: a != w, list_par[i])) #removing features with (+) weight

                    #if negative words found
                    elif top_dict[w] > 0: #originally "<
                        list_par[i] = list_par[i] + [w]*constant #add constant number of negative words into it

                else:

                    #if negative word isn't there then add it
                    if top_dict[w] > 0: #originally "<"
                        list_par[i] = list_par[i] + [w]*constant

    #create modified test text
    modified_data = "D:/MyStudy/Data Warehousing & Data Mining/Project for Data Warehousing/modified_data.txt"
    with open(modified_data, "w+") as moddata:

        #change list_par into list of string
        list_string = [' '.join(x) for x in list_par]

        for i in range(len(list_string)):
            moddata.write(list_string[i]+"\n")
        

    
        
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.

import os
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


def load_data():
    print("loading data...")
    import codecs


    ham_file_loaction = os.listdir("/home/alireza/Desktop/email spam with knn/data/ham")
    spam_file_loaction = os.listdir("/home/alireza/Desktop/email spam with knn/data/spam")


    data = []

    for file_name in ham_file_loaction:
        path = "/home/alireza/Desktop/email spam with knn/data/ham/" + file_name

        with codecs.open(path, 'r', encoding='utf-8',errors='ignore') as fdata:
            data.append([str(fdata.read()), "ham"])



    for file_name in spam_file_loaction:
        path = "/home/alireza/Desktop/email spam with knn/data/spam" + "/" + file_name

        with codecs.open(path, 'r', encoding='utf-8',errors='ignore') as fdata:
            data.append([str(fdata.read()), "spam"])


    data = np.array(data)
    print("flag 1 : load data -----> success")

    return data

#----------------------------------------------

def pre_processing(data):
    print("pre processing data ...")

    punctuation = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    stop_words = stopwords.words('english')

    for record in data:
        for item in punctuation:
            record[0] = record[0].replace(item, "")

            word_split = record[0].split()

        new_word = [word.lower() for word in word_split if word not in stop_words]

        record[0] = "".join(new_word)

    print("flag 2 : pre processing data  -----> success")

    return data

def split_data(data):
    print("Splitting data...")
        
    features = data[:, 0]   
    labels = data[:, 1]     
    training_data, test_data, training_labels, test_labels =\
    train_test_split(features, labels, test_size = 0.27, random_state = 42)
        
    print("flag 3: splitted data")
    return training_data, test_data, training_labels, test_labels


#------------------------------------------

def get_count(text):

    word_count = dict()
    words = text.split()
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

#----------------------------------------

def euclidean_difference(test_WordCounts, training_WordCounts):
    total = 0

    for word in test_WordCounts:
        if (word in test_WordCounts) and (word in training_WordCounts):
            total += (test_WordCounts[word] - training_WordCounts[word]) ** 2
            del training_WordCounts[word]
        else:
            total += test_WordCounts[word] **  2

    for word in training_WordCounts:
        total += training_WordCounts[word] **  2

    return total ** (0.5)

#----------------------------------------

def get_class(data):
    spam_count = 0
    ham_count = 0

    for element in data:
        if element[0] == "spam":
            spam_count += 1
        else:
            ham_count += 1

    if spam_count > ham_count:
        return "spam"
    else:
        return "ham"


#---------------------------------------

def knn_classifier(training_data, training_labels, test_data, K):
    print("Running KNN Classifier...")

    result = []

    training_data = [get_count(text_training) for text_training in training_data]

    for text in test_data:
        similarity = []

        test_Wordcount = get_count(text)

        for index in range(0 , len(training_data)):
            distance = euclidean_difference(test_Wordcount, training_data[index])

            similarity.append([training_labels[index] , distance])


        similarity = sorted(similarity, key = lambda item : item[1])[ : K]


        result.append(get_class(similarity))
    print("KNN finish...")

    return result
#---------------------------------------

def accuracy(predict, test_labels):
    return ((predict == test_labels).sum()) / len(predict)

#---------------------------------------

def detail_model(predict, test_labels):
    correct = (predict == test_labels).sum()
    wrong = len(predict) - correct
    return correct, wrong

#---------------------------------------

def main(K):

    data = load_data()
    data = pre_processing(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    size = len(test_data)
    result = knn_classifier(training_data, training_labels, test_data, K)

    print(f"training data size  :  {len(training_data)}")
    print(f"test data size  :  {len(test_data)}")
    print(f"accuracy of model :  {accuracy(result, test_labels) * 100} % ")
    print(f"Number correct : {detail_model(result, test_labels)[0]}")
    print(f"Number wrong : {detail_model(result, test_labels)[1]}")




main(11)
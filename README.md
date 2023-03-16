# email spam with KNN 


## What is KNN?

KNN (K-Nearest Neighbours) is one of the very straightforward supervised learning algorithms. However, unlike the traditional supervised learning algorithms, such as Multinomial Naive Bayes algorithm, KNN doesn’t have an independent training stage, and then a stage where the labels for the test data are predicted based on the trained model. Rather, the features of every test data item are compared with the features of every training data item in real time, and then the K nearest training data items are selected, and the most frequent class among them is given to the test data item.

In the context of email classification (spam or ham), the features to be compared are the frequencies of words in each email. The euclidean distance is used to determine the similarity between two emails; the smaller the distance, the more similar. The Euclidean Distance formula used in the algorithm is as follows :

![image](https://user-images.githubusercontent.com/89921883/225516305-0e8e68f4-aec5-4346-9f03-af4f3ed8ba51.png)


Once the Euclidean Distance between a test email and each training email is calculated, the distances are sorted in ascending order (nearest to farthest), and the K-nearest neighbouring emails are selected. If the majority is spam, then the test email is labelled as spam, else, it is labelled as ham.


![image](https://user-images.githubusercontent.com/89921883/225516489-8a816046-8d9f-4a86-aaf9-3d037551b98d.png)

In the example shown above, K = 5; we are comparing the email we want to classify to the nearest 5 neighbours. In this case, 3 out of 5 emails are classified as ham (non-spam), and 2 are classified as spam. Therefore, the unknown email will be given the class of the majority: ham. Now that we have seen how KNN works, let’s move on to implementing the classifier using code!

## Implementation

To have a quick idea of what we’ll be coding in Python, it’s always a good practice to write pseudo code:

1. Load the spam and ham emails
2. Remove common punctuation and symbols
3. Lowercase all letters
4. Remove stopwords (very common words like pronouns, articles, etc.)
5. Split emails into training email and testing emails
6. For each test email, calculate the similarity between it and all training emails
        6.1.  For each word that exists in either test email or training email, count its frequency in both emails
        6.2.  calculate the euclidean distance between both emails to determine similarity
7. Sort the emails in ascending order of euclidean distance
8. Select the k nearest neighbors (shortest distance)
9. Assign the class which is most frequent in the selected k nearest neighbours to the new email



## The Data Set

The email data set for spam and ham (normal email) is obtained from “The Enron-Spam datasets”. It can be found at http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html under Enron2. The data set we’re using contains 5857 emails. Every email is stored in a text file, and the text files are divided and stored into two folders, ham folder, and spam folder. This means that the emails are already labelled. Every text file will be loaded by the program, and each email will be read and stored as a string variable. Every distinct word inside the string will be counted as a feature.


## result : 


![reesult of model](https://github.com/parvvaresh/email-spam-with-KNN-from-scratch-/blob/main/result.png)

---------------------------------------------

Follow me on virtual pages : 


[![Twitter Badge](https://img.shields.io/badge/-Twitter-1da1f2?style=flat-square&labelColor=1da1f2&logo=twitter&logoColor=white&link=https://twitter.com/Yaronzz)](https://twitter.com/parvvaresh)
[![Email Badge](https://img.shields.io/badge/-Email-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:yaronhuang@foxmail.com)](mailto:parvvaresh@gmail.com)
[![Instagram Badge](https://img.shields.io/badge/-Instagram-purple?style=flat&logo=instagram&logoColor=white&link=https://instagram.com/parvvaresh/)](https://space.bilibili.com/7708412)
[![Github Badge](https://img.shields.io/badge/-Github-232323?style=flat-square&logo=Github&logoColor=white&link=https://space.bilibili.com/7708412)](https://github.com/parvvaresh)

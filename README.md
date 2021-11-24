# Product-Rating-Based-on-Sentiment-of-Tweets

* Project was aimed at rating a product based on Sentiment of large number of tweets corresponding to that product. Natural Language Processing was used to extract features in tweets.
* Sentiment was determined using various Machine Learning, Deep Learning and Dictionary based algorithms, which were compared on basis of different evaluation parameters.
* Skills– Python, Big Data, Natural Language Processing, Machine Learning and Deep Learning

## Brief Project Flow

![alt text](https://github.com/sameer7483/Product-Rating-Based-on-Sentiment-of-Tweets/blob/master/Flow.JPG)

### Retrieving tweets content from twitter API
### Data Pre-processing:
  * Removal Non Printable characters
  * Removal of URLs
  * Escaping HTML characters
  * Split Attached words (DisplayIsAwesome -> Display Is Awesome)
  * Emoji Replacer
  * Replacing regular expressions (should've -> should have)
  * Removal of unnecessary punctuation and tags
  * Tokenization
  * Slang word Replacer (lol -> laugh out loud)
  * Replacing of Abbreviations (A.S.A.P. -> as soon as possible)
  * Replace repeating characters (happpppy -> happy)
  * Checking Spellings
  * Lemmatizing (eating -> eat)
  * Replacing words with Contraction (do not uglify -> do beautify)
  * Removal of Stop words
  * Removal of Non Dictionary words
  * Language Translation
    
### Text Feature Extraction:
  * Representation of Bag of words using n-gram
  * Normalizing and Weighting with diminishing importance of
    * Tokens that are present in most of the samples and documents.
    * Tokens with are very sparse
  * Term Frequency-Inverse Document Frequency (TF-IDF)
    * Term Frequency (TF) = (Number of times term t appears in a document)/(Number of terms in the document)
    * Inverse Document Frequency (IDF) = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
    * The IDF of a rare word is high, whereas the IDF of a frequent word is likely to be low. Thus having the effect of highlighting words that are distinct.
    * We calculate TF-IDF value of a term as = TF * IDF
  * Word2Vector to reconstruct linguistic context of words
  
### Algorithms for determining Sentiment of tweets:
  #### Supervised Learning 
  * Decision Tree
                                  
                                  start = time.clock()
                                  clf = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
                                  clf.fit(x_train,y_train)
                                  Time_dt = time.clock() - start

                                  with open('pickle/DecisionTreeClassifier.pickle','wb') as f:
                                   pickle.dump(clf,f)

                                  pickle_in = open('pickle/DecisionTreeClassifier.pickle','rb')
                                  clf = pickle.load(pickle_in)

                                  y_pred = clf.predict(x_test)
  * Random Forest

                                  start = time.clock()
                                  clf = RandomForestClassifier(n_estimators=100, random_state=0, min_samples_split=2, min_samples_leaf=1)
                                  clf.fit(x_train,y_train)
                                  Time_rf = time.clock() - start

                                  with open('pickle/RandomForestClassifier.pickle','wb') as f:
                                   pickle.dump(clf,f)

                                  pickle_in = open('pickle/RandomForestClassifier.pickle','rb')
                                  clf = pickle.load(pickle_in)

                                  y_pred = clf.predict(x_test)

  * Gaussian Naive Bayes
  
                                  start = time.clock()
                                  clf = GaussianNB()
                                  clf.fit(x_train,y_train)
                                  Time_gnb = time.clock() - start

                                  with open('pickle/GaussianNB.pickle','wb') as f:
                                   pickle.dump(clf,f)

                                  pickle_in = open('pickle/GaussianNB.pickle','rb')
                                  clf = pickle.load(pickle_in)

                                  y_pred = clf.predict(x_test)
                                  
  * Multinomial Naive Bayes

                                   start = time.clock()
                                   clf = MultinomialNB()
                                   clf.fit(x_train,y_train)
                                   Time_mnb = time.clock() - start

                                   with open('pickle/MultinomialNB.pickle','wb') as f:
                                    pickle.dump(clf,f)

                                   pickle_in = open('pickle/MultinomialNB.pickle','rb')
                                   clf = pickle.load(pickle_in)
                                   y_pred = clf.predict(x_test)
                                   
                                   
  * Support Vector Classification

                                   start = time.clock()
                                   clf = svm.SVC(kernel="linear", random_state = 0)
                                   clf.fit(x_train,y_train)
                                   Time_svc = time.clock() - start

                                   with open('pickle/SVC.pickle','wb') as f:
                                    pickle.dump(clf,f)

                                   pickle_in = open('pickle/SVC.pickle','rb')
                                   clf = pickle.load(pickle_in)

                                   y_pred = clf.predict(x_test)
                                   
  * Logistic Regression

                                   start = time.clock()
                                   clf = linear_model.LogisticRegression()
                                   clf.fit(x_train,y_train)
                                   Time_lr = time.clock() - start

                                   with open('pickle/LogisticRegression.pickle','wb') as f:
                                    pickle.dump(clf,f)

                                   pickle_in = open('pickle/LogisticRegression.pickle','rb')
                                   clf = pickle.load(pickle_in)
                                   y_pred = clf.predict(x_test)
                                   
  * Neural Network

                                   input_length = len(x_train[2])
                                   n_classes = 2

                                   Y_train = []
                                   Y_test = []


                                   for i in range(0,len(y_train)):
                                    try:
                                     if(n_classes == 3):
                                      if y_train[i] == 0:
                                       a = [0,1,0]
                                      elif y_train[i] == 1:
                                       a= [1,0,0]
                                      else:
                                       a = [0,0,1]
                                     else:
                                      if y_train[i] == 1:
                                       a = [1,0]
                                      else:
                                       a = [0,1]
                                    except:
                                     if(n_classes == 3):
                                      a = [1,0,0]
                                     else:
                                      a = [1,0]

                                    Y_train.append(a)


                                   for i in range(0,len(y_test)):
                                    try:
                                     if(n_classes == 3):
                                      if y_test[i] == 0:
                                       a = [0,1,0]
                                      elif y_test[i] == 1:
                                       a= [1,0,0]
                                      else:
                                       a = [0,0,1]
                                     else:
                                      if y_test[i] == 1:
                                       a = [1,0]
                                      else:
                                       a = [0,1]
                                    except:
                                     if(n_classes == 3):
                                      a = [1,0,0]
                                     else:
                                      a = [1,0]

                                    Y_test.append(a)
                                    
  * Multi Layer perceptron

                                    start = time.clock()
                                    clf = MLPClassifier(hidden_layer_sizes=(100,100,100,), random_state=1)
                                    clf.fit(x_train,y_train)
                                    Time_mlp = time.clock() - start

                                    with open('pickle/MLPClassifier.pickle','wb') as f:
                                     pickle.dump(clf,f)

                                    pickle_in = open('pickle/MLPClassifier.pickle','rb')
                                    clf = pickle.load(pickle_in)
                                    y_pred = clf.predict(x_test)
                                    
  ##### Lexicon Based
  * Dictionary based: Dictionary based sentiment analysis is based on comparison between the text or corpus with pre-established dictionaries of positive, negative and neutral words.
  * Dictionary based with Score: Sentiment score of a tweet is given by the sum of positive and negative ratings of words in it.
### Evaluation Parameters:
  * Accuracy
  * Precision
  * F1Score
  * Recall
  * Cohen Kappa
  * Hamming Loss
  * Jaccard Similarity
  * Execution Time
  
  
### Result Analysis:
Some Abbreviation:
• MNB – Multinomial Naïve Bayes
• SVM – Support Vector Machine
• LR – Logistic Regression
• DB – Dictionary Based
• DBS – Dictionary Based with Score

| Evaluation Parameters | MNB | SVM | LR | DB | DBS | Remark |
| --------------------- |:---:|:---:|:--:|:--:|:---:|:------:|
| Accuracy | 0.69 | 0.70 | 0.71 | 0.62 | 0.58 |Logistic Regression fits the data points as if they are along a continuous function.SVM fits a function (hyperplane) that attempts to separate two classes of data that could be of multiple dimensions.|
| Precision | 0.64 | 0.64 | 0.65 | 0.61 | 0.56 |Logistic Regression makes a prediction for the probability using a direct functional form where as Naive Bayes figures out how the data was generated given the results.|
| F1 Score | 0.71 | 0.72 | 0.73 | 0.57 | 0.64 |Decision trees in Random Forest algorithm chop up the feature space into rectangles (or in higher dimensions, hyper-rectangles). On other hand, logistic regression assumes that there is one smooth linear decision boundary.|
| Recall | 0.75 | 0.75 | 0.75 | 0.48 | 0.71 |For relatively small amount of linearly separable data to perform simple classification tasks, LR is a great, robust model.|
| Cohen kappa | 0.39 | 0.40 | 0.42 | 0.26 | 0.16 |SVM could have difficulty when the classes are not separable or there is not enough margin to fit a (n_dimensions - 1) hyperplane between the two classes whereas Logistic Regression fits the data points as if they are along a continuous function.|
| Hamming loss | 0.30 | 0.29 | 0.28 | 0.37 | 0.41 |Hamming Loss measures the hamming distance between the prediction of labels and the true label. Thus, smaller the value of Hamming Loss, more efficient is the algorithm.|
| Jaccard Similarity | 0.69 | 0.70 | 0.71 | 0.62 | 0.58 |SVM has difficulty in training the data which are not linearly separable. On other hand, logistic regression assumes that there is one smooth linear decision boundary..|
| Time | 0.17 | 350 | 0.15 | 4.31 | 6.07 |Execution time of different algorithms (in seconds)|

                  

### Application: Using Sentiment Classification model to give ratings to various movies and products.

I used this sentiment classification model to give ratings to various movies and products. In today’s world where data has been growing at an exponential rate it can be a great tool to know about the sentiment of masses. I have used twitter tweets as data to know about sentiment of people for a particular product or Movies.
In this application when a keyword corresponding to a product or movie or any event is fed. Dynamically twitter tweets containing that key word starts to download. After enough data is fetched, it goes through series of data preprocessing operations and once we have obtained clean data then it is used by Machine Learning algorithm to come up with the rating corresponding to that product.

![alt text](https://github.com/sameer7483/Product-Rating-Based-on-Sentiment-of-Tweets/blob/master/iphone.JPG)












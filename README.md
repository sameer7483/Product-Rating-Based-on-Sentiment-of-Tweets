# Product-Rating-Based-on-Sentiment-of-Tweets

* Project was aimed at rating a product based on Sentiment of large number of tweets corresponding to that product. Natural Language Processing was used to extract features in tweets.
* Sentiment was determined using various Machine Learning, Deep Learning and Dictionary based algorithms, which were compared on basis of different evaluation parameters.
* Skills– Python, Big Data, Natural Language Processing, Machine Learning and Deep Learning

## Brief Project Flow

* Retrieving tweets content from twitter API
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

                           tweet = filter(lambda x: x in printable, row['tweet'])
                           #print tweet

                           a = re.sub(r"http\S+", "", tweet)   # Removing URLs
                           a = html_parser.unescape(a)					   # Removing HTML characters
                           a = RegexpReplacer().replace(a) 
                           a = a.lower()            

                           print a

                           word_tokens = word_tokenize(a)                      # Tokenization
                           stop_words = set(stopwords.words('english'))

                           s = []
                           for a in word_tokens:
                            a = slang_replacer.replace(a)
                            s.append(a)

                           s = " ".join(s)
                           s = str(s).translate(None, string.punctuation)   

                           word_tokens = word_tokenize(s) 
                           s = []
                           for a in word_tokens:
                            a = RepeatReplacer().replace(a)			
                            a = a.split(" ")
                            for i in a:
                             i = SpellingReplacer().replace(i)
                             i = i.split(" ")
                             for j in i:
                              j = lemmatizer.lemmatize(j)
                              s.append(j)

                           print s
                           s = AntonymReplacer().replace_negations(s)

                           p = []
                           for a in s:
                            if a not in stop_words:
                             if a.isalpha():
                              p.append(a)	 

                           p = " ".join(w for w in p if w in words)
                           p = re.sub(r'\b\w{1,2}\b', '', p)
                           a = word_tokenize(p)
                           length = len(a)

                           p = " ".join(a)
                           row['tweet'] = p

                           if length >= 3:
                            d = {'senti': [row['senti']], 'tweet': [row['tweet']]}
                            df = pd.DataFrame(data=d)
                            with open(output_file, 'a') as f:
                             df.to_csv(f, header=False)
    
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

                           def Evaluation_parameters(y_test, y_pred, name):
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = average_precision_score(y_test, y_pred)
                            f1score = f1_score(y_test, y_pred)
                            recall = recall_score(y_test, y_pred)
                            cohen_kappa = cohen_kappa_score(y_test, y_pred)
                            Hamming_loss = hamming_loss(y_test, y_pred)
                            jaccard_similarity = jaccard_similarity_score(y_test, y_pred)
                            Confusion_matrix = confusion_matrix(y_test, y_pred)

                            plt.figure()
                            plot_confusion_matrix(Confusion_matrix, classes=["Positive", "Negative"], title='Confusion matrix of ' + name)
                            plt.show()

                            return accuracy, precision, f1score, recall, cohen_kappa, Hamming_loss, jaccard_similarity, Confusion_matrix

### Application: Using Sentiment Classification model to give ratings to various movies and products.












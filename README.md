# Product-Rating-Based-on-Sentiment-of-Tweets

* Project was aimed at rating a product based on Sentiment of large number of tweets corresponding to that product. Natural Language Processing was used to extract features in tweets.
* Sentiment was determined using various Machine Learning, Deep Learning and Dictionary based algorithms, which were compared on basis of different evaluation parameters.
* Skills– Python, Big Data, Natural Language Processing, Machine Learning and Deep Learning

## Brief Project Flow

* Retrieving tweets content from twitter API
* Data Pre-processing:
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
* Text Feature Extraction:
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

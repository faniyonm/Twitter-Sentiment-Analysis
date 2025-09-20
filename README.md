
# Twitter Sentiment Analysis

Twitter Sentiment Analysis is the process of using Python to understand the emotions or opinions expressed in tweets automatically. By analyzing the text we can classify tweets as positive, negative or neutral. This helps businesses and researchers track public mood, brand reputation or reactions to events in real time. Python libraries like TextBlob, Tweepy and NLTK make it easy to collect tweets, process the text and perform sentiment analysis efficiently.

## Load Data
The Sentiment140 dataset is a large collection of 1.6 million tweets labeled for sentiment analysis. It provides a foundation for training and evaluating models that classify tweets as positive, negative, or neutral. Researchers and businesses use it to study public opinion, brand perception, and social trends at scale. With Python libraries like Pandas, and scikit-learn, we can easily load, clean, and analyze this dataset to build sentiment analysis models efficiently.
The Sentiment140 dataset is accessed using Pandas which allows us to directly load the dataset from a CSV file into a DataFrame. We keep only the polarity column (which shows the sentiment label: 0 for negative, 2 for neutral, 4 for positive) and the tweet text column (which contains the tweet content).

## Remove Neutral Tweets
Here we remove neutral tweets where polarity is 2, map the labels so 0 stays negative and 4 becomes 1 for positive. Then we print how many positive and negative tweets are left in the data.

## Lowercasing
We create a simple function to convert all text to lowercase for consistency, apply it to every tweet in the dataset, and then display the original and cleaned versions of the first few tweets.

## Preparing the Data for Modeling
### Train-Test Split
We split the `clean_text` and `polarity` columns into training and testing sets using an 80/20 split. The parameter `random_state=42` is used to ensure reproducibility.

### Perform Vectorization
We create a TF-IDF vectorizer that converts text into numerical features using unigrams and bigrams, limited to 5000 features. It is fitted and transformed on the training data, then applied to the test data. Finally, we print the shapes of the resulting TF-IDF matrices.

## Model Training
We will test different machine learning models on the data to evaluate their performance and identify which one works best for sentiment classification.

### Logistic Regression
We train a Logistic Regression model with up to 100 iterations on the TF-IDF features. The model then predicts sentiment labels for the test data, and we print the accuracy along with a detailed classification report for evaluation.

**Result Summary:**  
The Logistic Regression model achieved an accuracy of 79.5%. The classification report shows balanced performance, with precision, recall, and F1-scores around 0.80 for both negative and positive classes, indicating the model is effective at distinguishing sentiment in tweets.

### Bernoulli Naive Bayes
We train a Bernoulli Naive Bayes classifier on the TF-IDF features from the training data. The model then predicts sentiments for the test data, and we print the accuracy along with a detailed classification report for evaluation.

**Result Summary:**  
The Bernoulli Naive Bayes model achieved an accuracy of 76.6%. The classification report shows fairly balanced performance, with precision, recall, and F1-scores around 0.76–0.77 for both classes, indicating the model performs reasonably well but slightly below Logistic Regression.

### Support Vector Machine (SVM)
We train a Support Vector Machine (SVM) model with a maximum of 1000 iterations on the TF-IDF features. The model then predicts sentiment labels for the test data, and we print the accuracy along with a detailed classification report to evaluate its performance.

**Result Summary:**  
The Support Vector Machine (SVM) model achieved an accuracy of 79.5%. The classification report shows balanced precision, recall, and F1-scores around 0.79–0.80 for both negative and positive classes, indicating performance comparable to Logistic Regression.

## Sample Predictions
Three sample tweets are taken and transformed into TF-IDF features using the same vectorizer. These features are then passed to the trained BernoulliNB, SVM, and Logistic Regression models to predict sentiment. The predictions are printed for each classifier, where 1 represents Positive and 0 represents Negative.

**Sample Predictions:**  
- Logistic Regression: [1 0 1]  
- BernoulliNB: [1 0 1]  
- SVM: [1 0 1]  

All three models — Logistic Regression, Bernoulli Naive Bayes, and SVM — predicted the same results for the sample tweets: [1 0 1], meaning the first and third tweets were classified as Positive and the second tweet as Negative. We can see that our models are working fine and giving the same predictions even with different approaches.


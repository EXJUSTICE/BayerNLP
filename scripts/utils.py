import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import Counter
import warnings; warnings.simplefilter('ignore')
import numpy as np
import string
from nltk import ngrams
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from xgboost import XGBClassifier
from lightgbm import LGBMModel,LGBMClassifier, plot_importance
from sklearn.metrics import confusion_matrix, accuracy_score,balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, roc_curve, auc


def sentiment(text):
    # Sentiment polarity of the reviews
    pol = []
    for i in text:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)
    return pol

def clean(sentence): 
    # changing to lower case
    lower = sentence.str.lower()
    
    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    
    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe


def sentiment(text):
    # Sentiment polarity of the reviews
    pol = []
    for i in text:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)
    return pol

def processing(data):
    data['clean'] =clean(data['Sentence'])
    stop_words = set(stopwords.words('english'))
    data['clean'] = data['clean'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    # Can also stem words
    Snow_ball = SnowballStemmer("english")
    data['clean_stem'] = data['clean'].apply(lambda x: " ".join(Snow_ball.stem(word) for word in x.split()))
    # Various polarity definitions using Textblob from NLTK, as a baseline - will need to calculate metrics lataer
    data['polarity'] = sentiment(data['Sentence'])
    data['polarity_clean']=sentiment(data['clean'])
    data['polarity_clean_stem']=sentiment(data['clean_stem'])
    #Create Features for use in LightGBM /XGBOOST

    #Word count in each review
    data['count_word']=data["clean_stem"].apply(lambda x: len(str(x).split()))

    #Unique word count 
    data['count_unique_word']=data["clean_stem"].apply(lambda x: len(set(str(x).split())))

    #Letter count
    data['count_letters']=data["Sentence"].apply(lambda x: len(str(x)))

    #punctuation count
    data["count_punctuations"] = data["Sentence"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    #upper case words count
    data["count_words_upper"] = data["Sentence"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    #title case words count
    data["count_words_title"] = data["Sentence"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    #Number of stopwords
    data["count_stopwords"] = data["Sentence"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

    #Average length of the words
    data["mean_word_len"] = data["clean_stem"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    #drop the NaN
    data =data.drop(columns=["check","ID"])
    return data

#Define general purpose hyperparam CV function
#Note that GridSearchCV auto f the estimator is a classifier and y is either binary or multiclass, StratifiedKFold 
def hyperparam_tune(clf, params,X_train,y_train,X_test,y_test):
    if clf == "lgbm":
        clf = LGBMClassifier(seed=42)
    elif clf == "xgb":
        clf = XGBClassifier()
    elif clf == "lr":
         clf =LogisticRegression()
    elif clf == "svc":
        clf = LinearSVC()
    elif clf =="rf":
        clf = RandomForestClassifier()
    elif clf =="mlp":
        clf= MLPClassifier()
    elif clf =="gb":
        clf = GradientBoostingClassifier
        
    search = GridSearchCV(clf,params,scoring='balanced_accuracy') #Handling imbalance, average of recall obtained on each class.
    search.fit(X=X_train, y=y_train)
    best = search.best_params_
    predicted = search.predict(X_test)
    accuracy =accuracy_score(y_test, predicted)
    accuracy_b =balanced_accuracy_score(y_test, predicted)
    f1 =f1_score(y_test, predicted,average="weighted") #Weighted chosen to account for imbalance
    report = classification_report(y_test, predicted)
    
    return best,accuracy,accuracy_b,f1,report, search





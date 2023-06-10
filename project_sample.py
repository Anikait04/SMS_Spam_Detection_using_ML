#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Extract data

# In[2]:


#data is of encoding Windows-1252
import chardet
with open('spam.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# In[3]:


df = pd.read_csv('spam.csv',encoding='Windows-1252') 


# In[4]:


df['v2'][0]


# In[5]:


df.shape


# # Data Cleaning

# In[6]:


df.info()


# In[7]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


df.sample(5)


# In[9]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[11]:


df['target'] = encoder.fit_transform(df['target'])


# In[12]:


df.head()


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df = df.drop_duplicates(keep='first')


# In[16]:


df.shape


# # EDA

# In[17]:


df.head()


# In[18]:


df['target'].value_counts()


# In[19]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[20]:


#Data is imbalanced


# In[21]:


import nltk


# In[22]:


nltk.download('punkt')


# In[23]:


df['text'].apply(len)


# In[24]:


df['num_characters'] = df['text'].apply(len)


# In[25]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df[['num_characters','num_words','num_sentences']].describe()


# In[30]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[31]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[32]:


import seaborn as sns


# In[33]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[34]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[35]:


sns.pairplot(df,hue='target')


# In[36]:


sns.heatmap(df.corr(),annot=True)


# # Data Preprocessing

# In[37]:


nltk.download('stopwords')


# In[38]:


from nltk.corpus import stopwords


# In[39]:


import string
string.punctuation


# In[40]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[41]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[42]:


transform_text("I loved the lectures in class about Machine Learning. What about you?")


# In[43]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[44]:


df.head()


# In[45]:


pip install wordcloud


# In[46]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[47]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[48]:


plt.imshow(spam_wc)


# In[49]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[50]:


plt.imshow(ham_wc)


# In[51]:


df.head()


# In[52]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        


# In[53]:


len(spam_corpus)


# In[54]:


from collections import Counter
plt.bar(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[55]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[56]:


len(ham_corpus)


# In[57]:


from collections import Counter
plt.bar(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[58]:


df.head()


# In[59]:


df['text'][2]


# # Model Building

# In[60]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[61]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[62]:


X.shape


# In[63]:


y = df['target'].values


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[66]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[67]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[68]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[69]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[70]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[71]:


# we used tfidf to transform the data and we used Multinomial Naive Bayes MNB


# In[72]:


get_ipython().system('pip install xgboost')


# In[73]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[74]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[75]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[76]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[77]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[78]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[79]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[80]:


performance_df


# In[81]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[82]:


performance_df1


# In[83]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[84]:


df.head()


# In[85]:


data=df['text'][2]
data


# In[86]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[87]:


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


# In[88]:


input_sms="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"


# In[89]:


transformed_sms=transform_text(input_sms)


# In[90]:


vector_input=tfidf.transform([transformed_sms])


# In[91]:


result=model.predict(vector_input)
if result==1:
    print("Spam")
else:
    print("Ham")


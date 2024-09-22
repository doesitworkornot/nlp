from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

data_path = './dataset/sib200/data/rus_Cyrl/'
train_path = data_path + 'train.tsv'
test_path = data_path + 'test.tsv'

df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')
print(df.head())

text = df['text']
vectorizer = CountVectorizer()
vectorizer.fit(text)
print(f'Lenght of dictionary: {len(sorted(vectorizer.vocabulary_))}')

X = vectorizer.transform(df['text'])
LE = LabelEncoder()
y = LE.fit_transform(df['category'])
clf = LogisticRegression(random_state=0).fit(X, y)

X_test = vectorizer.transform(test_df['text'])
y_test = LE.transform(test_df['category'])

y_pred = clf.predict(X_test)
acc = clf.score(X_test, y_test)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f'Scores is \nACC: {acc:.3f}\nF1M: {f1:.3f}')
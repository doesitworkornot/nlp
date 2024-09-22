import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


morph = pymorphy2.MorphAnalyzer()


def normalize_text(text):
    words = text.split()
    normalized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(normalized_words)


data_path = './dataset/sib200/data/rus_Cyrl/'
train_path = data_path + 'train.tsv'
test_path = data_path + 'test.tsv'
corp_path = './dataset/ruwiki_pymorphy.csv'

df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')
print(df.columns)
print(df.head())

wiki_df = pd.read_csv(corp_path, sep='\t')
print(wiki_df.columns)
print(wiki_df.head())

text_normalized = df['text'].apply(normalize_text)
print('Normalizing wiki corp')
tf_train_normalized = wiki_df[',proc_sentence']
print('Training tf-idf')
vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 1))
vectorizer.fit(tf_train_normalized)
print(f'Lenght of dictionary: {len(sorted(vectorizer.vocabulary_))}')


X = vectorizer.transform(text_normalized)
LE = LabelEncoder()
y = LE.fit_transform(df['category'])
clf = LogisticRegression(random_state=0).fit(X, y)


test_text_normalized = test_df['text'].apply(normalize_text)
X_test = vectorizer.transform(test_text_normalized)
y_test = LE.transform(test_df['category'])

y_pred = clf.predict(X_test)
acc = clf.score(X_test, y_test)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f'Scores is \nACC: {acc:.3f}\nF1M: {f1:.3f}')
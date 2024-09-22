import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import numpy as np


morph = pymorphy2.MorphAnalyzer()


def hyperparameter_search_catboost(X, y, n_iter=50, cv=3, random_state=42):
    # Define the parameter grid for CatBoost
    param_dist = {
        'iterations': np.arange(450, 550, 2),
        'depth': np.arange(6, 9),
        'learning_rate': np.linspace(0.01, 0.3, 10),
        'l2_leaf_reg': np.linspace(1, 10, 5),
        'border_count': np.arange(32, 256, 32),
        'random_strength': np.linspace(0, 1, 5),
        'bagging_temperature': np.linspace(0, 1, 5),
        'one_hot_max_size': [2, 5, 10, 20],
        'rsm': np.linspace(0.5, 1, 6),
    }

    # Initialize CatBoost classifier
    clf = CatBoostClassifier(loss_function='MultiClass', verbose=0)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       scoring='accuracy',
                                       cv=cv,
                                       verbose=1,
                                       random_state=random_state,
                                       n_jobs=-1)

    # Perform the random search
    random_search.fit(X, y)

    # Print the best parameters and best score
    print("Best Parameters: ", random_search.best_params_)
    print("Best Accuracy: ", random_search.best_score_)

    # Return the best estimator
    return random_search.best_estimator_


def normalize_text(text):
    words = text.split()
    normalized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(normalized_words)


data_path = './dataset/sib200/data/rus_Cyrl/'
train_path = data_path + 'train.tsv'
test_path = data_path + 'test.tsv'

df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')
print(df.head())


text_normalized = df['text'].apply(normalize_text)

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 1))
vectorizer.fit(text_normalized)
# vectorizer = CountVectorizer()
# vectorizer.fit(text_normalized)
print(f'Lenght of dictionary: {len(sorted(vectorizer.vocabulary_))}')


X = vectorizer.transform(text_normalized)

LE = LabelEncoder()
y = LE.fit_transform(df['category'])
best_clf = hyperparameter_search_catboost(X, y)


test_text_normalized = test_df['text'].apply(normalize_text)
X_test = vectorizer.transform(test_text_normalized)
y_test = LE.transform(test_df['category'])

y_pred = best_clf.predict(X_test)
acc = best_clf.score(X_test, y_test)

f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f'Scores is \nACC: {acc:.3f}\nF1M: {f1:.3f}')
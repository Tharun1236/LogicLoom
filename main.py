!pip install xgboost -q

import pandas as pd
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Loading the training and testing data saved in Drive
df_train = pd.read_csv('/content/drive/My Drive/Hazards_LABELLED_TRAIN.csv')
df_test = pd.read_csv('/content/drive/My Drive/Hazards_UNLABELLED_TEST.csv')

# Clean text
df_train['clean_text'] = df_train['text'].apply(clean_text)
df_test['clean_text'] = df_test['text'].apply(clean_text)

# Encoding the hazard labels
label_encoder = LabelEncoder()
df_train['label_encoded'] = label_encoder.fit_transform(df_train['hazard-type'])

# Features and labels
X = df_train['clean_text']
y = df_train['label_encoded']
X_test_final = df_test['clean_text']

# Training and Validation Splitting
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# The pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', XGBClassifier(eval_metric='mlogloss'))
])

# Hyperparameter grid
param_grid = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [4, 6],
    'clf__learning_rate': [0.1, 0.2],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0]
}

# Class weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Randomized Search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    scoring='f1_macro',
    n_iter=15,
    cv=2,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Training with the sample weights
search.fit(X_train, y_train, clf__sample_weight=sample_weights)

# Evaluating
y_val_pred = search.best_estimator_.predict(X_val)

# We tried to search for the best hyperparameters
print("✅ Best Params:", search.best_params_)

# Classification report
print("\n📋 Validation Classification Report:\n")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

# Macro F1 Score
macro_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"\n⭐ Macro F1 Score on Validation Set: {macro_f1:.4f}")

# Prediction on test set
y_test_pred = search.best_estimator_.predict(X_test_final)
predicted_labels = label_encoder.inverse_transform(y_test_pred)

# Save predictions directly as a .csv file
df_test['hazard'] = predicted_labels
df_test[['ID', 'hazard']].to_csv('/content/hazards_final.csv', index=False)

print("✅ Predictions saved to hazards_final.csv")

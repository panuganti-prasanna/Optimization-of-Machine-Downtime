import pandas as pd 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV

scaled_data = pd.read_csv("C:/Users/ADMIN/Desktop/Project/scaled_data")

Y_encoded = pd.read_csv("C:/Users/ADMIN/Desktop/Project/project_final/project_main/encoded_data.csv")

# Assuming X and y are your features and target variable
X_train, X_test, y_train, y_test = train_test_split(scaled_data ,Y_encoded , test_size=0.2, random_state=42)


# Define a simpler parameter grid with different values
param_grid = {
    'n_estimators': [25],
    'learning_rate': [0.06],
    'max_depth': [3],
    'min_samples_split': [5],
    'min_samples_leaf': [2]
}

# Create the Gradient Boosting classifier
gb = GradientBoostingClassifier(random_state=42)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(gb, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_gb = grid_search.best_estimator_

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(best_gb, X_train, y_train, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Fit the best model on the training data
best_gb.fit(X_train, y_train)

# Make predictions on the test data
gb_y_pred = best_gb.predict(X_test)

# Make predictions on the train data 
gb_y_pred_t = best_gb.predict(X_train)

# Calculate evaluation metrics
gb_accuracy_t = accuracy_score(y_train, gb_y_pred_t)

print("Accuracy_train", gb_accuracy_t)

# Calculate evaluation metrics
gb_accuracy = accuracy_score(y_test, gb_y_pred)
gb_precision = precision_score(y_test, gb_y_pred)
gb_recall = recall_score(y_test, gb_y_pred)
gb_f1 = f1_score(y_test, gb_y_pred)
gb_roc_auc = roc_auc_score(y_test, best_gb.predict_proba(X_test)[:, 1])

# Print the best hyperparameters and evaluation metrics
print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy:", gb_accuracy)
print("Precision:", gb_precision)
print("Recall:", gb_recall)
print("F1 Score:", gb_f1)
print("ROC AUC Score:", gb_roc_auc)

pickle.dump( best_gb, open('Gradient_boosting.pkl', 'wb'))




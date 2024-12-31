
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score , roc_curve, auc
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose




file_path = 'rapido_case_study_dataset.csv'
data = pd.read_csv(file_path)

# Check the first few rows of the dataset
print(data.head(10))


# Assuming 'Cancellation_Status' is the target column
X = data.drop('Cancellation_Status', axis=1)  # Drop target column
y = data['Cancellation_Status']  # Target variable

# Drop rows with missing values in both X and y to ensure alignment
data_clean = data.dropna(subset=['Cancellation_Status'])  # Drop rows with missing target values

# Now, split the data again
X = data_clean.drop('Cancellation_Status', axis=1)  # Features
y = data_clean['Cancellation_Status']  # Target variable



# Reset index after dropping rows to ensure alignment
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Check if there are non-numeric columns and encode them if necessary
non_numeric_columns = X.select_dtypes(exclude=['number']).columns
for column in non_numeric_columns:
    X[column] = LabelEncoder().fit_transform(X[column])

# Check the shapes to ensure alignment
print(X.shape)  # Should be (num_samples, num_features)
print(y.shape)  # Should be (num_samples,)

# Now, try the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Check the shapes of the split data
print(X_train.shape)
print(X_test.shape)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test, rf_predictions))

# Train a simple Neural Network
dl_model = Sequential()
dl_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
dl_model.add(Dense(32, activation='relu'))
dl_model.add(Dense(1, activation='sigmoid'))
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the DL Model
dl_loss, dl_accuracy = dl_model.evaluate(X_test, y_test)
print(f"Deep Learning Model Accuracy: {dl_accuracy}")

# Generate Insights
feature_importances = rf_model.feature_importances_
important_features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
print("Feature Importances from Random Forest:")
print(important_features)

# Example Insight: Ride Fare Optimization
if 'Ride_Fare' in data.columns:
    # Ensure only numeric columns are considered for correlation
    numeric_data = data.select_dtypes(include=['number'])
    
    if 'Ride_Fare' in numeric_data.columns:
        # Compute correlations for numeric columns
        fare_correlation = numeric_data.corr()['Ride_Fare'].sort_values(ascending=False)
        print("Correlation of features with Ride_Fare:")
        print(fare_correlation)
    else:
        print("'Ride_Fare' column is not numeric or missing after preprocessing.")
else:
    print("'Ride_Fare' column does not exist in the dataset.")

# Visualize important features
important_features.head(10).plot(kind='bar')
plt.title('Top 10 Important Features')
plt.show()


# Assuming 'Cancellation_Status' is the target variable
if 'Cancellation_Status' in data.columns:
    # Train a Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Feature importance
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importance from Random Forest:")
    print(feature_importance)

    # Visualize top features
    feature_importance.head(10).plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title('Top 10 Important Features for Cancellation Prediction')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()

    # Evaluate the model
    rf_predictions = rf_model.predict(X_test)
    rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    print(f"Random Forest ROC AUC Score: {rf_roc_auc}")
    print("Classification Report:")
    print(classification_report(y_test, rf_predictions))
else:
    print("'Cancellation_Status' column not found in the dataset.")


# Build a simple Neural Network model
dl_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = dl_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
dl_loss, dl_accuracy = dl_model.evaluate(X_test, y_test)
print(f"Deep Learning Model Accuracy: {dl_accuracy}")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('DL Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('DL Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Calculate ROC curve for Random Forest
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

# Calculate ROC curve for Deep Learning model
dl_probs = dl_model.predict(X_test).ravel()
dl_fpr, dl_tpr, _ = roc_curve(y_test, dl_probs)
dl_auc = auc(dl_fpr, dl_tpr)

# Plot ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})', color='blue')
plt.plot(dl_fpr, dl_tpr, label=f'Deep Learning (AUC = {dl_auc:.2f})', color='red')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('Comparison of ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()


import shap

# Explain Random Forest predictions with SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot of feature importance
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)

# Explanation for a single prediction
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0, :])




# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Visualize clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('Customer Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
 

# Analyze correlation of fare with cancellations
correlation_matrix = data.corr()
print("Correlation with Ride_Fare:")
print(correlation_matrix['Ride_Fare'].sort_values(ascending=False))

# Visualize fare vs. cancellations
sns.boxplot(x='Cancellation_Status', y='Ride_Fare', data=data)
plt.title('Ride Fare vs. Cancellation Status')
plt.show()


# Decompose time series (example: daily average ride fare)
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    daily_avg_fare = data['Ride_Fare'].resample('D').mean()
    decomposition = seasonal_decompose(daily_avg_fare, model='additive', period=30)

    # Plot decomposition
    decomposition.plot()
    plt.show()

import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='customer_cancellation'
)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Optimal number of units: Layer 1: {best_hps.get('units_1')}, Layer 2: {best_hps.get('units_2')}")



# Build LSTM model for fare prediction
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=20, batch_size=32)

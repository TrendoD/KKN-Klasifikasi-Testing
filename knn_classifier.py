import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess data
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Convert Y/T to 1/0 for columns C1-C6 and C11-C14
    categorical_columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C11', 'C12', 'C13', 'C14']
    for col in categorical_columns:
        processed_df[col] = processed_df[col].map({'Y': 1, 'T': 0})
    
    # Convert percentages to decimals for C7 and C9
    processed_df['C7'] = processed_df['C7'].str.rstrip('%').astype(float) / 100
    processed_df['C9'] = processed_df['C9'].str.rstrip('%').astype(float) / 100
    
    # C8 and C10 are already decimal but need to handle comma as decimal separator
    processed_df['C8'] = processed_df['C8'].astype(str).str.replace(',', '.').astype(float)
    processed_df['C10'] = processed_df['C10'].astype(str).str.replace(',', '.').astype(float)
    
    return processed_df

# Load training data
print("Loading and processing training data...")
training_data = pd.read_csv('dataset Covid.csv')
training_processed = preprocess_data(training_data)

# Load data to be classified
print("Loading and processing new data...")
new_data = pd.read_csv('dataset baru.csv')
new_processed = preprocess_data(new_data)

# Save the processed training data
training_processed.to_csv('training_data_processed.csv', index=False)

# Save the processed new data
new_processed.to_csv('new_data_processed.csv', index=False)

# Extract features and target for training
X_train = training_processed[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']]
y_train = training_processed['Zona']

# Extract features for new data
X_new = new_processed[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']]

# Normalize features using Min-Max scaling
print("Normalizing features...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_new_scaled = scaler.transform(X_new)

# Using fixed k=33 as requested
k = 33
print(f"Using K={k} for the KNN classifier as requested...")

# Train KNN model with k=33
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Get class distribution
class_distribution = y_train.value_counts()
print("\nDistribusi Kelas pada Data Training:")
for zona, count in class_distribution.items():
    print(f"Zona {zona}: {count} data ({count/len(y_train)*100:.2f}%)")

# Predict zones for new data
print("\nPredicting zones for new data...")
predictions = knn.predict(X_new_scaled)

# Get prediction probabilities for new data
prediction_probs = knn.predict_proba(X_new_scaled)
class_indices = {class_name: i for i, class_name in enumerate(knn.classes_)}

# Add predictions to new data
new_processed['Zona_Prediksi'] = predictions

# Add confidence score (probability) for each prediction
for zona in knn.classes_:
    zona_idx = class_indices[zona]
    new_processed[f'Prob_{zona}'] = prediction_probs[:, zona_idx]

# Save the final results
new_processed.to_csv('hasil_klasifikasi_detail_k33.csv', index=False)

# Display the classification results with probability
print("\nHasil Klasifikasi Detail (K=33):")
print("------------------------------")
for i, (idx, row) in enumerate(new_processed.iterrows()):
    print(f"Daerah {row['Daerah']}: {predictions[i]}")
    print(f"  Probabilitas per Zona:")
    for zona in knn.classes_:
        prob = row[f'Prob_{zona}']
        print(f"    - {zona}: {prob:.4f} ({prob*100:.2f}%)")
    print("  Nearest Neighbors:")
    
    # Get indices of nearest neighbors (showing first 10 for brevity)
    distances, indices = knn.kneighbors([X_new_scaled[i]], n_neighbors=k)
    for j, (neighbor_idx, distance) in enumerate(zip(indices[0], distances[0])):
        if j < 10:  # Show only first 10 neighbors to keep output manageable
            neighbor_daerah = training_processed.iloc[neighbor_idx]['Daerah']
            neighbor_zona = training_processed.iloc[neighbor_idx]['Zona']
            print(f"    - {j+1}. Daerah {neighbor_daerah} (Zona {neighbor_zona}), jarak: {distance:.4f}")
    if k > 10:
        print(f"    - ... dan {k-10} tetangga lainnya")
    print()

print("\nHasil klasifikasi tersimpan di file 'hasil_klasifikasi_detail_k33.csv'")
print("Data training yang sudah diproses tersimpan di 'training_data_processed.csv'")
print("Data baru yang sudah diproses tersimpan di 'new_data_processed.csv'") 
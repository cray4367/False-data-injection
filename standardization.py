import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuration
false_data_ratio = 0.15
test_size = 0.3
random_seed = 42

# Load real data
real_df = pd.read_csv("data.csv")

# Assign label 0 to legit data if not already present
if 'Label' not in real_df.columns:
    real_df['Label'] = 0

# Separate features and labels
X_real = real_df.drop(columns=['Label']).values
y_real = real_df['Label'].values

# Standardize the real data
scaler = StandardScaler()
X_real_std = scaler.fit_transform(X_real)

# Generate false data
n_false = int(false_data_ratio * len(X_real_std))
false_data = X_real_std[:n_false] + np.random.normal(0, 2, size=X_real_std[:n_false].shape)
false_labels = np.ones(n_false)

# Combine real and false data
X_combined = np.vstack((X_real_std, false_data))
y_combined = np.hstack((y_real, false_labels))

# Shuffle
np.random.seed(random_seed)
indices = np.arange(len(X_combined))
np.random.shuffle(indices)
X_combined = X_combined[indices]
y_combined = y_combined[indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=test_size, random_state=random_seed)

# Save to CSV
train_df = pd.DataFrame(X_train, columns=[f"Feature_{i+1}" for i in range(X_train.shape[1])])
train_df['Label'] = y_train
train_df.to_csv('train_standardized.csv', index=False)

test_df = pd.DataFrame(X_test, columns=[f"Feature_{i+1}" for i in range(X_test.shape[1])])
test_df['Label'] = y_test
test_df.to_csv('test_standardized.csv', index=False)

print("Standardized dataset with injected false data generated successfully!")
print("Train set: train_standardized.csv")
print("Test set: test_standardized.csv")

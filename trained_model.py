import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\DELL\Downloads\bank_dataset_cleaned.csv", encoding='latin1')

# -----------------------------
# 2. Encode Categorical Columns
# -----------------------------
target = "y"  # your target column name

for col in df.select_dtypes(include=["object"]).columns:
    if col != target:  # encode all object columns except target
        df[col] = LabelEncoder().fit_transform(df[col])

# Encode target if categorical
if df[target].dtype == "object":
    df[target] = LabelEncoder().fit_transform(df[target])

# -----------------------------
# 3. Features and Target
# -----------------------------
X = df.drop(target, axis=1).values.astype(float)
y = df[target].values.astype(float)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Build ANN Model
# -----------------------------
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # since "y" is binary yes/no
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 6. Train Model
# -----------------------------
ann_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# -----------------------------
# 7. Save Model
# -----------------------------
ann_model.save("ann_model.h5")
print("✅ Model trained and saved as ann_model.h5")

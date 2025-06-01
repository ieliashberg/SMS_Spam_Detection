# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers, callbacks, optimizers, Input

# Import utility modules
from data_utils import load_data, preprocess_text, add_length_features
from visualization_utils import (
    plot_label_distribution,
    plot_bar_metric,
    plot_correlation_matrix,
    show_top_words,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_cpu_usage
)
from model_utils import (
    train_svm,
    train_random_forest,
    train_naive_bayes_grid_search
)

# CPUTracker should be imported at the top so it's available for all sections
from cpu_tracker import CPUTracker

# ----------------------------------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# ----------------------------------------------------------------------------

PARQUET_PATH = "hf://datasets/marcov/sms_spam_promptsource/data/train-00000-of-00001.parquet"

df = load_data(PARQUET_PATH)
print("Data loaded. Shape:", df.shape)

# Visualize label distribution
label_counts = df['label'].value_counts()
plot_label_distribution(label_counts)

# Add length features
df = add_length_features(df)

# Compare means by class and plot
metrics = {}
for feat in ['Length_of_SMS', 'words_in_sms', 'sentences_in_sms']:
    metrics[feat] = {
        'spam': df.loc[df['label'] == 1, feat].mean(),
        'ham': df.loc[df['label'] == 0, feat].mean()
    }

for feat, vals in metrics.items():
    print(f"  {feat}: Spam={vals['spam']:.2f}, Ham={vals['ham']:.2f}")
    plot_bar_metric(
        vals['spam'],
        vals['ham'],
        title=f"Average {feat.replace('_', ' ')} for Spam vs. Ham",
        y_label=f"Avg {feat.replace('_', ' ')}"
    )

# Correlation heatmap
plot_correlation_matrix(df, ['Length_of_SMS', 'words_in_sms', 'sentences_in_sms'])

# ----------------------------------------------------------------------------
# 2. TEXT PREPROCESSING
# ----------------------------------------------------------------------------

# Create a new column "processed_sms" by applying preprocess_text
df['processed_sms'] = df['sms'].apply(preprocess_text)

# Show top words for spam and non-spam
show_top_words(df.loc[df['label'] == 1, 'processed_sms'], "Spam SMS")
show_top_words(df.loc[df['label'] == 0, 'processed_sms'], "Non-Spam SMS")

# ----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING & SPLIT
# ----------------------------------------------------------------------------

# Encode labels (0 = ham, 1 = spam)
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['processed_sms']).toarray()
y = df['label_enc']

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------
# 4. MODEL TRAINING & EVALUATION
# ----------------------------------------------------------------------------

# 4.1. SVM
svm_results = train_svm(X_train, y_train, X_test, y_test)
print(f"SVM Accuracy: {svm_results['accuracy']:.4f}")
plot_confusion_matrix(
    y_test,
    svm_results['model'].predict(X_test),
    labels=['Ham', 'Spam'],
    title="Confusion Matrix: SVM"
)
plot_roc_curve(
    y_test,
    svm_results['y_prob'],
    title="ROC Curve: SVM"
)
plot_cpu_usage(
    svm_results['timestamps'],
    svm_results['cpu_usage'],
    title="CPU Usage During SVM Training"
)

# 4.2. Random Forest
rf_results = train_random_forest(X_train, y_train, X_test, y_test)
print(f"Random Forest Accuracy: {rf_results['accuracy']:.4f}")
plot_confusion_matrix(
    y_test,
    rf_results['model'].predict(X_test),
    labels=['Ham', 'Spam'],
    title="Confusion Matrix: Random Forest",
    cmap='Greens'
)
plot_roc_curve(
    y_test,
    rf_results['y_prob'],
    title="ROC Curve: Random Forest"
)
plot_cpu_usage(
    rf_results['timestamps'],
    rf_results['cpu_usage'],
    title="CPU Usage During Random Forest Training"
)

# 4.3. Naïve Bayes Grid Search
nb_results = train_naive_bayes_grid_search(X_train, y_train, X_test, y_test)
print(
    f"Best NB Accuracy: {nb_results['accuracy']:.4f} "
    f"(alpha={nb_results['best_params']['alpha']}, fit_prior={nb_results['best_params']['fit_prior']})"
)
plot_confusion_matrix(
    y_test,
    nb_results['model'].predict(X_test),
    labels=['Ham', 'Spam'],
    title="Confusion Matrix: Best Naïve Bayes",
    cmap='Purples'
)
plot_roc_curve(
    y_test,
    nb_results['y_prob'],
    title="ROC Curve: Naïve Bayes"
)
plot_cpu_usage(
    nb_results['timestamps'],
    nb_results['cpu_usage'],
    title="CPU Usage During Naïve Bayes Grid Search"
)

# ----------------------------------------------------------------------------
# 5. NEURAL NETWORKS
# ----------------------------------------------------------------------------

# 5.1. Basic Neural Network
basic_nn = Sequential([
    Input(shape=(X_train.shape[1],)),       # ← explicit Input layer
    layers.Dense(128, activation='relu'),   # ← no more input_shape here
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
basic_nn.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Track CPU usage for the basic NN
nn_tracker = CPUTracker()
nn_tracker.start()
history_basic = basic_nn.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
basic_timestamps, basic_cpu_usage = nn_tracker.stop()

# Evaluate basic NN on the test set
test_loss, test_acc = basic_nn.evaluate(X_test, y_test, verbose=0)
print(f"Basic NN Test Accuracy: {test_acc:.4f}")

# Plot training vs. validation loss for basic NN
plt.figure(figsize=(6, 4))
plt.plot(history_basic.history['loss'], label='train_loss')
plt.plot(history_basic.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Basic NN: Training vs. Validation Loss')
plt.legend()
plt.show()

plot_cpu_usage(
    basic_timestamps,
    basic_cpu_usage,
    title="CPU Usage During Basic NN Training"
)

# 5.2. Regularized Neural Network
reg_nn = Sequential([
    Input(shape=(X_train.shape[1],)),       # ← explicit Input layer
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
reg_nn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

nn_tracker2 = CPUTracker()
nn_tracker2.start()
history_reg = reg_nn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)
reg_timestamps, reg_cpu_usage = nn_tracker2.stop()

# Evaluate regularized NN on the test set
test_loss_reg, test_acc_reg = reg_nn.evaluate(X_test, y_test, verbose=0)
print(f"Regularized NN Test Accuracy: {test_acc_reg:.4f}")

# Plot training vs. validation loss for regularized NN
plt.figure(figsize=(6, 4))
plt.plot(history_reg.history['loss'], label='train_loss')
plt.plot(history_reg.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Regularized NN: Training vs. Validation Loss')
plt.legend()
plt.show()

plot_cpu_usage(
    reg_timestamps,
    reg_cpu_usage,
    title="CPU Usage During Regularized NN Training"
)

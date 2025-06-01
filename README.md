
# SMS Spam Detection Overview
- **cpu_tracker.py**  
  Defines a `CPUTracker` class that measures CPU usage over time in a background thread.

- **data_utils.py**  
  Contains functions to load the Parquet dataset from Hugging Face and perform text-length feature extraction, as well as a `preprocess_text(...)` function for tokenization, stop-word removal, and stemming.

- **visualization_utils.py**  
  Provides plotting routines:  
  - Pie chart of label distribution  
  - Bar plots for average text lengths  
  - Correlation heatmap of length features  
  - Word-cloud and bar graphs for top words  
  - Confusion matrix heatmap  
  - ROC curve plot  
  - CPU usage plot  

- **model_utils.py**  
  Implements training functions for:  
  - **train_svm(...)**: Linear SVM with CPU tracking and ROC computation  
  - **train_random_forest(...)**: Random Forest classifier with CPU tracking and ROC computation  
  - **train_naive_bayes_grid_search(...)**: Multinomial Naïve Bayes over a grid (alpha ∈ [0.1, 1.0], fit_prior = True/False) with CPU tracking, returning the best model and ROC data  

- **main.py**  
  Orchestrates the full pipeline:  
  1. Load & preprocess data  
  2. Add length features and visualize them  
  3. Text preprocessing & top-word visualizations  
  4. Train/test split  
  5. Model training (SVM, RF, NB), evaluation, and plotting  
  6. Two feedforward neural networks (basic + regularized) with CPU usage tracking  

- **requirements.txt**  
  Lists all required Python packages. Use this to create a reproducible virtual environment.

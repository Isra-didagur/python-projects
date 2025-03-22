import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK resource download failed. Make sure NLTK is installed correctly.")

# Set random seed for reproducibility
np.random.seed(42)

# Function to load data - replace with your data loading code
def load_data(file_path='easy_ham'):
   
    if file_path:
        try:
            # Attempt to load the user's CSV file
            df = pd.read_csv(file_path='easy_ham')
            
            # Check if the required columns exist or need to be renamed
            if 'text' not in df.columns and 'email' in df.columns:
                df.rename(columns={'email': 'text'}, inplace=True)
            if 'label' not in df.columns and 'spam' in df.columns:
                df.rename(columns={'spam': 'label'}, inplace=True)
                
            print(f"Loaded data from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Using sample data instead.")
    
    # Generate simple sample data for demonstration
    print("Using sample demonstration data.")
    texts = [
        "Congratulations! You've won a free vacation to Hawaii. Call now to claim your prize!",
        "Meeting scheduled for tomorrow at 10 AM in the conference room.",
        "URGENT: Your account has been compromised. Click here to reset your password immediately.",
        "Hi John, can you send me the quarterly report when you get a chance?",
        "FREE FREE FREE VIAGRA! Best prices online guaranteed! Buy now!",
        "Reminder: Your dental appointment is scheduled for Friday at 2 PM.",
        "You are the 1,000,000th visitor! Claim your $1000 prize now!",
        "Project update: We've completed milestone 2 ahead of schedule.",
        "URGENT: Your payment of $349.99 has been processed. Contact us if this was not authorized.",
        "Team lunch is planned for Friday. Please let me know if you can attend."
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam
    
    return pd.DataFrame({'text': texts, 'label': labels})

# Text preprocessing functions
def clean_text(text):
   
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text, remove_stopwords=True, stemming=False, lemmatization=True):
  
    if not text:
        return []
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Apply stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Apply lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def preprocess_data(df, text_column='text', label_column='label'):
   
    # Copy the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Clean text
    print("Cleaning text...")
    df_processed['cleaned_text'] = df_processed[text_column].apply(clean_text)
    
    # Tokenize and join back tokens for later use with vectorizers
    print("Tokenizing text...")
    df_processed['processed_text'] = df_processed['cleaned_text'].apply(
        lambda x: ' '.join(tokenize_text(x))
    )
    
    # Display examples of original vs processed text
    print("\nText preprocessing examples:")
    examples = df_processed[[text_column, 'processed_text', label_column]].head(3)
    for i, row in examples.iterrows():
        print(f"\nOriginal: {row[text_column]}")
        print(f"Processed: {row['processed_text']}")
        print(f"Label: {'Spam' if row[label_column] == 1 else 'Not Spam'}")
    
    return df_processed

# Exploratory Data Analysis functions
def perform_eda(df, text_column='text', label_column='label'):
  
    print("\n--- Exploratory Data Analysis ---")
    
    # Basic dataset info
    print(f"\nDataset shape: {df.shape}")
    print(f"\nLabel distribution:")
    label_counts = df[label_column].value_counts()
    print(label_counts)
    
    # Calculate and display percentage of spam vs. not spam
    spam_percentage = label_counts.get(1, 0) / len(df) * 100
    nonspam_percentage = label_counts.get(0, 0) / len(df) * 100
    print(f"\nSpam: {spam_percentage:.2f}%")
    print(f"Not Spam: {nonspam_percentage:.2f}%")
    
    # Visualize label distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df[label_column])
    plt.title('Distribution of Spam vs. Not Spam')
    plt.xlabel('Label (1 = Spam, 0 = Not Spam)')
    plt.ylabel('Count')
    plt.show()
    
    # Calculate and plot email length distribution by class
    df['text_length'] = df[text_column].apply(lambda x: len(str(x)))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', hue=label_column, bins=50, kde=True)
    plt.title('Email Length Distribution by Class')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Calculate and display basic statistics about email length
    print("\nEmail length statistics:")
    length_stats = df.groupby(label_column)['text_length'].describe()
    print(length_stats)
    
    # Word count analysis
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=label_column, y='word_count', data=df)
    plt.title('Word Count by Email Class')
    plt.xlabel('Label (1 = Spam, 0 = Not Spam)')
    plt.ylabel('Word Count')
    plt.show()
    
    return df

# Feature extraction
def extract_features(X_train, X_test, method='tfidf', max_features=5000, ngram_range=(1, 2)):
    
    print(f"\nExtracting features using {method.upper()}...")
    
    if method.lower() == 'bow':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
    else:  # Default to TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
    
    # Fit and transform on training data
    X_train_features = vectorizer.fit_transform(X_train)
    
    # Transform test data
    X_test_features = vectorizer.transform(X_test)
    
    print(f"Feature extraction complete. Shape of training features: {X_train_features.shape}")
    
    # Display top features (most frequent or highest TF-IDF score)
    feature_names = vectorizer.get_feature_names_out()
    if method.lower() == 'bow':
        feature_counts = X_train_features.sum(axis=0).A1
        top_indices = feature_counts.argsort()[-20:][::-1]
    else:
        feature_sums = X_train_features.sum(axis=0).A1
        top_indices = feature_sums.argsort()[-20:][::-1]
    
    print("\nTop 20 features:")
    for i, idx in enumerate(top_indices, 1):
        if method.lower() == 'bow':
            print(f"{i}. {feature_names[idx]} (Count: {feature_counts[idx]})")
        else:
            print(f"{i}. {feature_names[idx]} (TF-IDF Sum: {feature_sums[idx]:.2f})")
    
    return X_train_features, X_test_features, vectorizer

# Model training and evaluation
def train_evaluate_model(X_train, X_test, y_train, y_test, model_type='naive_bayes'):
   
    print(f"\n--- Training {model_type.replace('_', ' ').title()} Model ---")
    
    # Select model based on type
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        print(f"Unknown model type: {model_type}. Using Naive Bayes as default.")
        model = MultinomialNB()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate and display metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Display detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
    
    # Generate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Spam', 'Spam'], 
                yticklabels=['Not Spam', 'Spam'])
    plt.title(f'Confusion Matrix - {model_type.replace("_", " ").title()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # If the model supports probabilities, display ROC curve
    if hasattr(model, "predict_proba"):
        from sklearn.metrics import roc_curve, auc
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    return model

# Function to find most predictive features
def analyze_feature_importance(vectorizer, model, class_names=['Not Spam', 'Spam']):
   
    print("\n--- Feature Importance Analysis ---")
    
    feature_names = vectorizer.get_feature_names_out()
    
    # For Naive Bayes
    if isinstance(model, MultinomialNB):
        # Get log probabilities for each class
        feature_log_probs = model.feature_log_prob_
        
        # For each class
        for i, class_name in enumerate(class_names):
            # Sort indices by log probability
            sorted_indices = feature_log_probs[i].argsort()
            
            # Get top 20 features with highest log probability for this class
            top_indices = sorted_indices[-20:]
            
            print(f"\nTop 20 features for {class_name}:")
            for j, idx in enumerate(reversed(top_indices), 1):
                print(f"{j}. {feature_names[idx]} (Log Prob: {feature_log_probs[i][idx]:.4f})")
    
    # For Logistic Regression and similar linear models
    elif hasattr(model, 'coef_'):
        # Get coefficients
        coefficients = model.coef_[0]
        
        # Sort by absolute coefficient value
        sorted_indices = np.argsort(np.abs(coefficients))
        
        # Get top 20 features with highest absolute coefficient
        top_indices = sorted_indices[-20:]
        
        print("\nTop 20 most important features:")
        for j, idx in enumerate(reversed(top_indices), 1):
            coef = coefficients[idx]
            print(f"{j}. {feature_names[idx]} (Coef: {coef:.4f}, Class: {'Spam' if coef > 0 else 'Not Spam'})")
            
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(20), coefficients[top_indices][::-1])
        plt.yticks(range(20), [feature_names[idx] for idx in top_indices][::-1])
        plt.xlabel('Coefficient Value')
        plt.title('Top 20 Feature Importance')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.show()
    
    # For tree-based models like Random Forest
    elif hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort by importance
        sorted_indices = np.argsort(importances)
        
        # Get top 20 features with highest importance
        top_indices = sorted_indices[-20:]
        
        print("\nTop 20 most important features:")
        for j, idx in enumerate(reversed(top_indices), 1):
            print(f"{j}. {feature_names[idx]} (Importance: {importances[idx]:.4f})")
            
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(20), importances[top_indices][::-1])
        plt.yticks(range(20), [feature_names[idx] for idx in top_indices][::-1])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance')
        plt.show()
    
    else:
        print("This model type doesn't support feature importance analysis.")

# Create pipeline to streamline the process
def create_pipeline(feature_extraction='tfidf', classifier='naive_bayes', max_features=5000):
   
    # Select vectorizer
    if feature_extraction.lower() == 'bow':
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
    else:  # Default to TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
    
    # Select classifier
    if classifier.lower() == 'naive_bayes':
        model = MultinomialNB()
    elif classifier.lower() == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif classifier.lower() == 'svm':
        model = SVC(probability=True, random_state=42)
    elif classifier.lower() == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = MultinomialNB()  # Default
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])
    
    return pipeline

# Hyperparameter tuning
def tune_hyperparameters(pipeline, X_train, y_train):
  
    print("\n--- Hyperparameter Tuning ---")
    
    # Define parameter grid based on the pipeline components
    param_grid = {}
    
    # Vectorizer parameters
    if isinstance(pipeline.named_steps['vectorizer'], CountVectorizer):
        param_grid.update({
            'vectorizer__max_features': [3000, 5000],
            'vectorizer__ngram_range': [(1, 1), (1, 2)]
        })
    else:
        param_grid.update({
            'vectorizer__max_features': [3000, 5000],
            'vectorizer__ngram_range': [(1, 1), (1, 2)]
        })
    
    # Classifier parameters
    if isinstance(pipeline.named_steps['classifier'], MultinomialNB):
        param_grid.update({
            'classifier__alpha': [0.1, 0.5, 1.0]
        })
    elif isinstance(pipeline.named_steps['classifier'], LogisticRegression):
        param_grid.update({
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        })
    elif isinstance(pipeline.named_steps['classifier'], SVC):
        param_grid.update({
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        })
    elif isinstance(pipeline.named_steps['classifier'], RandomForestClassifier):
        param_grid.update({
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20]
        })
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    # Perform grid search
    print("Performing grid search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Prediction function for new emails
def predict_spam(model, text, vectorizer=None):
  
    # Clean and preprocess the text
    cleaned_text = clean_text(text)
    processed_text = ' '.join(tokenize_text(cleaned_text))
    
    # Make prediction
    if isinstance(model, Pipeline) and 'vectorizer' in model.named_steps:
        # If using a pipeline, we don't need a separate vectorizer
        prediction = model.predict([processed_text])[0]
        probability = model.predict_proba([processed_text])[0][1]  # Probability of spam
    else:
        # If using separate model and vectorizer
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided when using a standalone model")
        
        # Transform text using the vectorizer
        text_features = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_features)[0]
        
        # Get probability if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(text_features)[0][1]  # Probability of spam
        else:
            probability = None
    
    return prediction, probability

# Main function to run the complete pipeline
def main(data_path=None, feature_method='tfidf', model_type='naive_bayes', use_pipeline=True):
    
    print("\n===== Email Spam Classification System =====\n")
    
  
    df = load_data(data_path)
    
    
    df_processed = preprocess_data(df)
    
    df_processed = perform_eda(df_processed)
    
    X = df_processed['processed_text']
    y = df_processed['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    if use_pipeline:
        
        print("\nUsing scikit-learn Pipeline for streamlined processing")
        
        pipeline = create_pipeline(feature_extraction=feature_method, classifier=model_type)
        
        print("\nTraining the model...")
        pipeline.fit(X_train, y_train)
        

        print("\nEvaluating the model:")
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
        
     
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Spam', 'Spam'], 
                    yticklabels=['Not Spam', 'Spam'])
        plt.title(f'Confusion Matrix - Pipeline ({feature_method.upper()} + {model_type.replace("_", " ").title()})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    
        analyze_feature_importance(pipeline.named_steps['vectorizer'], pipeline.named_steps['classifier'])
        
       
        print("\nExample predictions:")
        example_emails = [
            "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
            "Hi team, please review the attached report before tomorrow's meeting.",
            "URGENT: Your account has been suspended. Verify your information immediately."
        ]
        
        for email in example_emails:
            prediction, probability = predict_spam(pipeline, email)
            print(f"\nEmail: {email}")
            print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
            if probability is not None:
                print(f"Spam Probability: {probability:.4f}")
            print("-" * 50)
        
        model = pipeline  
    else:
       
        X_train_features, X_test_features, vectorizer = extract_features(
            X_train, X_test, method=feature_method
        )
        
        
        model = train_evaluate_model(X_train_features, X_test_features, y_train, y_test, model_type)
        
        analyze_feature_importance(vectorizer, model)
        
      
        print("\nExample predictions:")
        example_emails = [
            "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
            "Hi team, please review the attached report before tomorrow's meeting.",
            "URGENT: Your account has been suspended. Verify your information immediately."
        ]
        
        for email in example_emails:
            prediction, probability = predict_spam(model, email, vectorizer)
            print(f"\nEmail: {email}")
            print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
            if probability is not None:
                print(f"Spam Probability: {probability:.4f}")
            print("-" * 50)
    


    
    print("\n===== Email Spam Classification Complete =====")
    return model


if __name__ == "__main__":
   
    main(feature_method='tfidf', model_type='naive_bayes', use_pipeline=True)
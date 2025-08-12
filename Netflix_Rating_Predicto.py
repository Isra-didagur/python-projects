#rateflix that rates movies based on ml models ml pipelines and randomforest 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)


movies_df = pd.read_csv('Netflix_Dataset_Movie.csv')

# Display basic information about the dataset
print("Dataset shape:", movies_df.shape)
print("\nDataset information:")
print(movies_df.info())
print("\nSample data:")
print(movies_df.head())

# Data preprocessing and exploratory analysis
def explore_data(df):
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
   
    
    # If there's a year column, plot rating trends over time
    if 'year' in df.columns:
        plt.figure(figsize=(12, 6))
        year_ratings = df.groupby('year')['rating'].mean()
        plt.plot(year_ratings.index, year_ratings.values)
        plt.title('Average Rating by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Rating')
        plt.grid(True)
        plt.show()
    
    return df

# Call the exploratory function
movies_df = explore_data(movies_df)

# Feature Engineering
def engineer_features(df):
    """
    This function creates new features that might be useful for prediction.
    You should adapt this function based on the columns available in your dataset.
    """
    feature_df = df.copy()
    
    
    # Return the DataFrame with new features
    return feature_df

# Apply feature engineering
features_df = engineer_features(movies_df)

# Prepare data for model training
def prepare_training_data(df, target_col='rating'):
    """
    Prepares the data for training by separating features and target,
    and identifying numerical and categorical columns.
    """
    # Make a copy to avoid modifying the original
    df_model = df.copy()
    
    # Drop columns that are not useful for prediction (modify as needed)
    # Example: IDs, URLs, timestamps, etc.
    columns_to_drop = []
    for col in df_model.columns:
        if col.lower() in ['id', 'movie_id', 'timestamp', 'url', 'name', 'title']:
            columns_to_drop.append(col)
    
    # Drop target column from columns_to_drop if it was mistakenly added
    if target_col in columns_to_drop:
        columns_to_drop.remove(target_col)
    
    # Drop specified columns if they exist
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_model.columns]
    if existing_cols_to_drop:
        df_model = df_model.drop(columns=existing_cols_to_drop)
    
    # Separate features and target
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Identify text columns (for potential text feature extraction)
    text_cols = []
    for col in categorical_cols:
        # Check if the column contains long text (more than 100 characters on average)
        if X[col].dropna().astype(str).str.len().mean() > 100:
            text_cols.append(col)
            categorical_cols.remove(col)
    
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Text columns: {text_cols}")
    
    return X, y, numerical_cols, categorical_cols, text_cols

# Prepare training data
X, y, numerical_cols, categorical_cols, text_cols = prepare_training_data(features_df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build preprocessing pipeline
def create_pipeline(numerical_cols, categorical_cols, text_cols):
    """
    Creates a preprocessing and model pipeline with appropriate transformers for each column type.
    """
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Text preprocessing (if needed)
    text_transformers = []
    for text_col in text_cols:
        text_transformers.append((
            f'text_{text_col}',
            Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))
            ]),
            [text_col]
        ))
    
    # Combine all preprocessors
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ] + text_transformers
    
    # Remove any transformers with empty column lists
    transformers = [t for t in transformers if t[2]]
    
    # Create the full pipeline
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Create and return the full pipeline with the model
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        return model_pipeline
    else:
        # If no transformers are available, return a simple pipeline
        return Pipeline([
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

# Create pipeline
pipeline = create_pipeline(numerical_cols, categorical_cols, text_cols)

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Evaluate the model
def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluates the model and returns performance metrics.
    """
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Actual vs Predicted Movie Ratings')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.grid(True)
    plt.show()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    
    return mse, rmse, mae, r2

# Evaluate the model
mse, rmse, mae, r2 = evaluate_model(pipeline, X_test, y_test)

# Feature importance analysis
def analyze_feature_importance(pipeline, feature_names):
    """
    Extracts and visualizes feature importance from the model.
    """
    # Get feature names from column transformer
    if hasattr(pipeline['preprocessor'], 'transformers_'):
        all_features = []
        for name, trans, cols in pipeline['preprocessor'].transformers_:
            if name == 'cat' and hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                # Get the categorical feature names after one-hot encoding
                cat_features = []
                for i, col in enumerate(cols):
                    categories = trans.named_steps['onehot'].categories_[i]
                    for category in categories:
                        cat_features.append(f"{col}_{category}")
                all_features.extend(cat_features)
            elif name.startswith('text_'):
                # Get the text feature names (top words from TF-IDF)
                col = cols[0]
                if hasattr(trans, 'named_steps') and 'tfidf' in trans.named_steps:
                    vocab = trans.named_steps['tfidf'].get_feature_names_out()
                    text_features = [f"{col}_{word}" for word in vocab]
                    all_features.extend(text_features)
            else:
                # Add numerical features directly
                all_features.extend(cols)
        
        feature_names = all_features
    else:
        feature_names = feature_names
    
    # Extract feature importances
    if hasattr(pipeline['model'], 'feature_importances_'):
        importances = pipeline['model'].feature_importances_
        
        # Create DataFrame for better visualization
        if len(feature_names) == len(importances):
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        else:
            print("Error: Feature names length doesn't match importance array length")
            return None
    else:
        print("Model doesn't provide feature importances")
        return None


try:
    feature_importance = analyze_feature_importance(pipeline, X.columns)
    if feature_importance is not None:
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
except Exception as e:
    print(f"Could not analyze feature importance: {e}")

# Model tuning (optional - uncomment to run)
def tune_model(pipeline, X_train, y_train):
  
    print("Tuning the model (this may take some time)...")
    
    # Define parameter grid
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", -grid_search.best_score_)
    
    return grid_search.best_estimator_


# Function to predict ratings for new movies
def predict_rating(pipeline, new_data):
  
    # Ensure new_data has the same columns as the training data
    required_columns = list(X.columns)
    missing_columns = [col for col in required_columns if col not in new_data.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in new data: {missing_columns}")
        # Add missing columns with NaN values
        for col in missing_columns:
            new_data[col] = np.nan
    
    # Reorder columns to match training data
    new_data = new_data[required_columns]
    
    # Make predictions
    predictions = pipeline.predict(new_data)
    
    # Add predictions to the DataFrame
    result_df = new_data.copy()
    result_df['predicted_rating'] = predictions
    
    return result_df



print("\nMovie Rating Prediction Model Complete!")

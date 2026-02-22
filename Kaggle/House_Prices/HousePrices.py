
"""
Kaggle House Prices Competition - Complete Pipeline
This script handles the entire workflow from data loading to submission creation.
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Boosting libraries
import xgboost as xgb
import lightgbm as lgb
import joblib

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("All libraries imported successfully!\n")

# ============================================================================
# PHASE 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("="*80)
print("PHASE 1: DATA LOADING AND INITIAL EXPLORATION")
print("="*80)

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['Id']

print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}\n")

# Basic info
print("Training Data Info:")
print(train.info())

print("\n" + "="*80)
print("Target Variable (SalePrice) Statistics:")
print("="*80)
print(train['SalePrice'].describe())

# Check missing values
print("\n" + "="*80)
print("Missing Values in Training Data (Top 20):")
print("="*80)
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(20))
print(f"\nTotal missing percentage: {(train.isnull().sum().sum() / (train.shape[0] * train.shape[1])) * 100:.2f}%")

# Visualize target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train['SalePrice'], bins=50, edgecolor='black')
axes[0].set_xlabel('Sale Price')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Sale Prices')
axes[0].axvline(train['SalePrice'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${train["SalePrice"].mean():,.0f}')
axes[0].axvline(train['SalePrice'].median(), color='green', linestyle='--', 
                label=f'Median: ${train["SalePrice"].median():,.0f}')
axes[0].legend()

stats.probplot(train['SalePrice'], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot of Sale Prices')

plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSkewness: {train['SalePrice'].skew():.3f}")
print(f"Kurtosis: {train['SalePrice'].kurt():.3f}")

# ============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Correlation analysis - only numeric columns
numeric_train = train.select_dtypes(include=[np.number])
corr_matrix = numeric_train.corr()
corr_with_price = corr_matrix['SalePrice'].sort_values(ascending=False)

print("\nTop 15 Features Correlated with SalePrice:")
print(corr_with_price.head(15))

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
top_features = corr_with_price.head(11).index
sns.heatmap(train[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
plt.title('Correlation Heatmap - Top Features')
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualize key relationships
top_numeric = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_numeric):
    axes[idx].scatter(train[feature], train['SalePrice'], alpha=0.5)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('SalePrice')
    axes[idx].set_title(f'{feature} vs SalePrice')
    
    z = np.polyfit(train[feature], train['SalePrice'], 1)
    p = np.poly1d(z)
    axes[idx].plot(train[feature], p(train[feature]), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig('03_key_relationships.png', dpi=150, bbox_inches='tight')
plt.show()

# Identify outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(train['GrLivArea'], train['SalePrice'], alpha=0.5)
axes[0].set_xlabel('GrLivArea (Above Ground Living Area)')
axes[0].set_ylabel('SalePrice')
axes[0].set_title('GrLivArea vs SalePrice - Notice Outliers')

axes[1].scatter(train['LotArea'], train['SalePrice'], alpha=0.5)
axes[1].set_xlabel('LotArea')
axes[1].set_ylabel('SalePrice')
axes[1].set_title('LotArea vs SalePrice')

plt.tight_layout()
plt.savefig('04_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PHASE 3: DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*80)

# Simple numeric-only approach (compatible with current models)
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Id')
if 'SalePrice' in numeric_cols:
    numeric_cols.remove('SalePrice')

train_data = train[numeric_cols].fillna(train[numeric_cols].median())
X = train_data
y = train['SalePrice']

# Remove outliers
if 'GrLivArea' in X.columns:
    outlier_idx = (X['GrLivArea'] > 4000) & (y < 300000)
    X = X[~outlier_idx]
    y = y[~outlier_idx]

print(f"Final training set shape (numeric only): {X.shape}")
print(f"Target shape: {y.shape}\n")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# ============================================================================
# PHASE 3B: ENHANCED FEATURE ENGINEERING FOR PRE-TRAINED MODEL
# ============================================================================

print("\n[Building enhanced features for pre-trained model compatibility...]")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Prepare full dataset (train + test) for consistent preprocessing
train_full = train.copy()
test_full = test.copy()

# Get all columns except Id and SalePrice
feature_cols = [col for col in train_full.columns if col not in ['Id', 'SalePrice']]

# Separate categorical and numeric
cat_cols = train_full[feature_cols].select_dtypes(include='object').columns.tolist()
numeric_cols_full = train_full[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

# Handle missing values
train_prep = train_full.copy()
test_prep = test_full.copy()

for col in numeric_cols_full:
    train_prep[col] = train_prep[col].fillna(train_prep[col].median())
    test_prep[col] = test_prep[col].fillna(train_prep[col].median())

for col in cat_cols:
    mode_val = train_prep[col].mode()[0] if len(train_prep[col].mode()) > 0 else 'Unknown'
    train_prep[col] = train_prep[col].fillna(mode_val)
    test_prep[col] = test_prep[col].fillna(mode_val)

# One-hot encode categorical features
train_encoded = pd.get_dummies(train_prep[feature_cols], columns=cat_cols, drop_first=False)
test_encoded = pd.get_dummies(test_prep[feature_cols], columns=cat_cols, drop_first=False)

# Align test columns to match train
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# PCA on numeric features
pca = PCA(n_components=4, random_state=42)
pca_features = pca.fit_transform(train_encoded[numeric_cols_full])
pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(4)], index=train_encoded.index)

test_pca = pca.transform(test_encoded[numeric_cols_full])
test_pca_df = pd.DataFrame(test_pca, columns=[f'PC{i+1}' for i in range(4)], index=test_encoded.index)

# K-means clustering (20 clusters)
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(train_encoded[numeric_cols_full])
clusters_test = kmeans.predict(test_encoded[numeric_cols_full])

# Calculate distances to each centroid
centroids = kmeans.cluster_centers_
cluster_distances_train = np.linalg.norm(train_encoded[numeric_cols_full].values[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
cluster_distances_test = np.linalg.norm(test_encoded[numeric_cols_full].values[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

cluster_df_train = pd.DataFrame(cluster_distances_train, columns=[f'Centroid_{i}' for i in range(20)], index=train_encoded.index)
cluster_df_test = pd.DataFrame(cluster_distances_test, columns=[f'Centroid_{i}' for i in range(20)], index=test_encoded.index)

# Combine all features
X_full_train = pd.concat([train_encoded, pca_df, cluster_df_train], axis=1)
X_full_test = pd.concat([test_encoded, test_pca_df, cluster_df_test], axis=1)

print(f"Enhanced feature set shape: {X_full_train.shape}")
print(f"Number of features: {X_full_train.shape[1]}")

# Create corresponding train/val split aligned with original
outlier_mask = ~((train_prep['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)) if 'GrLivArea' in train_prep.columns else np.ones(len(train_prep), dtype=bool)
y_full = train['SalePrice'][outlier_mask]
X_full_train_clean = X_full_train[outlier_mask].reset_index(drop=True)
y_full = y_full.reset_index(drop=True)

X_full_train_split, X_full_val_split, y_full_train, y_full_val = train_test_split(
    X_full_train_clean, y_full, test_size=0.2, random_state=42
)

print(f"Enhanced training set: {X_full_train_split.shape}")
print(f"Enhanced validation set: {X_full_val_split.shape}\n")

# ============================================================================
# PHASE 4: EXTENSIVE HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: EXTENSIVE HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*80)

results = {}

# ==================== 1. RIDGE REGRESSION ====================
print("\n[1/6] Tuning Ridge Regression...")
ridge_params = {
    'alpha': [10, 100, 1000],
    'solver': ['lsqr'],
}

ridge_grid = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=2,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=0
)

ridge_grid.fit(X_train, y_train)
ridge_pred = ridge_grid.predict(X_val)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
ridge_r2 = r2_score(y_val, ridge_pred)

results['Ridge'] = {
    'best_params': ridge_grid.best_params_,
    'best_score': ridge_grid.best_score_,
    'rmse': ridge_rmse,
    'r2': ridge_r2,
    'model': ridge_grid.best_estimator_
}

print(f"OK - Best params: {ridge_grid.best_params_}")
print(f"OK - Best CV score: {-ridge_grid.best_score_:,.0f}")
print(f"OK - Validation RMSE: {ridge_rmse:,.0f}")
print(f"OK - Validation R²: {ridge_r2:.4f}")

# ==================== 2. LASSO REGRESSION ====================
print("\n[2/6] Tuning Lasso Regression...")
lasso_params = {
    'alpha': [1, 10, 100],
    'max_iter': [5000],
}

lasso_grid = GridSearchCV(
    Lasso(random_state=42),
    lasso_params,
    cv=2,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=0
)

lasso_grid.fit(X_train, y_train)
lasso_pred = lasso_grid.predict(X_val)
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_pred))
lasso_r2 = r2_score(y_val, lasso_pred)

results['Lasso'] = {
    'best_params': lasso_grid.best_params_,
    'best_score': lasso_grid.best_score_,
    'rmse': lasso_rmse,
    'r2': lasso_r2,
    'model': lasso_grid.best_estimator_
}

print(f"OK - Best params: {lasso_grid.best_params_}")
print(f"OK - Best CV score: {-lasso_grid.best_score_:,.0f}")
print(f"OK - Validation RMSE: {lasso_rmse:,.0f}")
print(f"OK - Validation R²: {lasso_r2:.4f}")

# ==================== 3. ELASTIC NET ====================
print("\n[3/6] Tuning ElasticNet with GridSearchCV...")
elasticnet_params = {
    'alpha': [0.1, 1],
    'l1_ratio': [0.5, 0.7],
    'max_iter': [5000],
}

elasticnet_grid = GridSearchCV(
    ElasticNet(random_state=42),
    elasticnet_params,
    cv=2,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=0
)

elasticnet_grid.fit(X_train, y_train)
elasticnet_pred = elasticnet_grid.predict(X_val)
elasticnet_rmse = np.sqrt(mean_squared_error(y_val, elasticnet_pred))
elasticnet_r2 = r2_score(y_val, elasticnet_pred)

results['ElasticNet'] = {
    'best_params': elasticnet_grid.best_params_,
    'best_score': elasticnet_grid.best_score_,
    'rmse': elasticnet_rmse,
    'r2': elasticnet_r2,
    'model': elasticnet_grid.best_estimator_
}

print(f"OK - Best params: {elasticnet_grid.best_params_}")
print(f"OK - Best CV score: {-elasticnet_grid.best_score_:,.0f}")
print(f"OK - Validation RMSE: {elasticnet_rmse:,.0f}")
print(f"OK - Validation R²: {elasticnet_r2:.4f}")

# ==================== 4. RANDOM FOREST (RandomizedSearchCV) ====================
print("\n[4/6] Tuning Random Forest with RandomizedSearchCV...")

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2'],
}

rf_random = RandomizedSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=42),
    rf_params,
    n_iter=8,
    cv=2,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=0
)

rf_random.fit(X_train, y_train)
rf_pred = rf_random.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
rf_r2 = r2_score(y_val, rf_pred)

results['RandomForest'] = {
    'best_params': rf_random.best_params_,
    'best_score': rf_random.best_score_,
    'rmse': rf_rmse,
    'r2': rf_r2,
    'model': rf_random.best_estimator_
}

print(f"OK - Best params: {rf_random.best_params_}")
print(f"OK - Best CV score: {-rf_random.best_score_:,.0f}")
print(f"OK - Validation RMSE: {rf_rmse:,.0f}")
print(f"OK - Validation R²: {rf_r2:.4f}")

# ==================== 5. GRADIENT BOOSTING (GridSearchCV + RandomizedSearchCV) ====================
print("\n[5/6] Tuning Gradient Boosting with GridSearchCV...")

gb_params = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [5, 10],
    'subsample': [0.8, 1.0],
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params,
    cv=2,
    n_jobs=1,
    scoring='neg_mean_squared_error',
    verbose=0
)

gb_grid.fit(X_train, y_train)
gb_pred = gb_grid.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
gb_r2 = r2_score(y_val, gb_pred)

results['GradientBoosting'] = {
    'best_params': gb_grid.best_params_,
    'best_score': gb_grid.best_score_,
    'rmse': gb_rmse,
    'r2': gb_r2,
    'model': gb_grid.best_estimator_
}

print(f"OK - Best params: {gb_grid.best_params_}")
print(f"OK - Best CV score: {-gb_grid.best_score_:,.0f}")
print(f"OK - Validation RMSE: {gb_rmse:,.0f}")
print(f"OK - Validation R²: {gb_r2:.4f}")

# ==================== 6. XGBOOST (Load Pre-trained Model with Enhanced Features) ====================
print("\n[6/6] Loading Pre-trained XGBoost Model (with enhanced features)...")

xgb_pretrained_loaded = False
try:
    xgb_best_model = joblib.load('xgb_mod.model')
    print(f"OK - Pre-trained model loaded successfully")
    
    xgb_pred = xgb_best_model.predict(X_full_val_split)
    xgb_rmse = np.sqrt(mean_squared_error(y_full_val, xgb_pred))
    xgb_r2 = r2_score(y_full_val, xgb_pred)
    
    results['XGBoost'] = {
        'best_params': 'Pre-trained model (loaded with enhanced features)',
        'best_score': None,
        'rmse': xgb_rmse,
        'r2': xgb_r2,
        'model': xgb_best_model
    }
    
    print(f"OK - Validation RMSE: {xgb_rmse:,.0f}")
    print(f"OK - Validation R²: {xgb_r2:.4f}")
    xgb_pretrained_loaded = True
except Exception as e:
    print(f"ERROR - Failed to load pre-trained model with enhanced features: {str(e)[:200]}")
    print("Falling back to training XGBoost from scratch...")
    
    xgb_params = {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.5],
        'reg_alpha': [0, 1],
        'reg_lambda': [0.1, 1],
    }
    
    xgb_random = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        xgb_params,
        n_iter=8,
        cv=2,
        n_jobs=1,
        scoring='neg_mean_squared_error',
        random_state=42,
        verbose=0
    )
    
    xgb_random.fit(X_train, y_train)
    xgb_pred = xgb_random.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    xgb_r2 = r2_score(y_val, xgb_pred)
    
    results['XGBoost'] = {
        'best_params': xgb_random.best_params_,
        'best_score': xgb_random.best_score_,
        'rmse': xgb_rmse,
        'r2': xgb_r2,
        'model': xgb_random.best_estimator_
    }
    
    print(f"OK - Best params: {xgb_random.best_params_}")
    print(f"OK - Best CV score: {-xgb_random.best_score_:,.0f}")
    print(f"OK - Validation RMSE: {xgb_rmse:,.0f}")
    print(f"OK - Validation R²: {xgb_r2:.4f}")
    print(f"OK - Validation RMSE: {xgb_rmse:,.0f}")
    print(f"OK - Validation R²: {xgb_r2:.4f}")

# ============================================================================
# PHASE 5: MODEL COMPARISON AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("HYPERPARAMETER TUNING RESULTS SUMMARY")
print("="*80)

# Create results summary
summary_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Best CV Score': [abs(results[model]['best_score']) for model in results.keys()],
    'Validation RMSE': [results[model]['rmse'] for model in results.keys()],
    'Validation R²': [results[model]['r2'] for model in results.keys()],
})

summary_df = summary_df.sort_values('Validation RMSE')
print("\n" + summary_df.to_string(index=False))

# Identify best model
best_model_name = summary_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"Best Parameters: {results[best_model_name]['best_params']}")
print(f"Validation RMSE: ${results[best_model_name]['rmse']:,.0f}")
print(f"Validation R²: {results[best_model_name]['r2']:.4f}")

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE comparison
axes[0].barh(summary_df['Model'], summary_df['Validation RMSE'], color='steelblue')
axes[0].set_xlabel('Validation RMSE')
axes[0].set_title('Model Performance Comparison - RMSE')
axes[0].invert_yaxis()

# R² comparison
axes[1].barh(summary_df['Model'], summary_df['Validation R²'], color='darkgreen')
axes[1].set_xlabel('Validation R²')
axes[1].set_title('Model Performance Comparison - R²')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nOK - Hyperparameter tuning completed successfully!")

# ============================================================================
# PHASE 6: ADVANCED STACKING WITH META-LEARNER
# ============================================================================

print("\n" + "="*80)
print("PHASE 6: ADVANCED STACKING WITH META-LEARNER")
print("="*80)

# Define base learners from tuned models
base_learners = [
    ('ridge', results['Ridge']['model']),
    ('lasso', results['Lasso']['model']),
    ('elasticnet', results['ElasticNet']['model']),
    ('rf', results['RandomForest']['model']),
    ('gb', results['GradientBoosting']['model']),
    ('xgb', results['XGBoost']['model']),
]

print(f"\nBase learners: {[name for name, _ in base_learners]}")

# ==================== 1. STACKING WITH RIDGE META-LEARNER ====================
print("\n[1/4] Training StackingRegressor with Ridge Meta-Learner...")

stacking_ridge = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(alpha=100),
    cv=2,
    n_jobs=1
)

stacking_ridge.fit(X_train, y_train)
stacking_ridge_pred = stacking_ridge.predict(X_val)
stacking_ridge_rmse = np.sqrt(mean_squared_error(y_val, stacking_ridge_pred))
stacking_ridge_r2 = r2_score(y_val, stacking_ridge_pred)

print(f"OK - Validation RMSE: {stacking_ridge_rmse:,.0f}")
print(f"OK - Validation R²: {stacking_ridge_r2:.4f}")

# ==================== 2. STACKING WITH LASSO META-LEARNER ====================
print("\n[2/4] Training StackingRegressor with Lasso Meta-Learner...")

stacking_lasso = StackingRegressor(
    estimators=base_learners,
    final_estimator=Lasso(alpha=1.0),
    cv=2,
    n_jobs=1
)

stacking_lasso.fit(X_train, y_train)
stacking_lasso_pred = stacking_lasso.predict(X_val)
stacking_lasso_rmse = np.sqrt(mean_squared_error(y_val, stacking_lasso_pred))
stacking_lasso_r2 = r2_score(y_val, stacking_lasso_pred)

print(f"OK - Validation RMSE: {stacking_lasso_rmse:,.0f}")
print(f"OK - Validation R²: {stacking_lasso_r2:.4f}")

# ==================== 3. STACKING WITH GRADIENT BOOSTING META-LEARNER ====================
print("\n[3/4] Training StackingRegressor with Gradient Boosting Meta-Learner...")

stacking_gb = StackingRegressor(
    estimators=base_learners,
    final_estimator=GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    cv=2,
    n_jobs=1
)

stacking_gb.fit(X_train, y_train)
stacking_gb_pred = stacking_gb.predict(X_val)
stacking_gb_rmse = np.sqrt(mean_squared_error(y_val, stacking_gb_pred))
stacking_gb_r2 = r2_score(y_val, stacking_gb_pred)

print(f"OK - Validation RMSE: {stacking_gb_rmse:,.0f}")
print(f"OK - Validation R²: {stacking_gb_r2:.4f}")

# ==================== 4. TUNING STACKING META-LEARNER WITH GRIDSEARCH ====================
print("\n[4/4] Tuning Stacking with GridSearchCV Meta-Learner Search...")

# Create a stacking regressor with Ridge as initial meta-learner
base_stacking = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(),
    cv=5,
    n_jobs=-1
)

# Grid search for optimal meta-learner parameters
meta_params = {
    'final_estimator__alpha': [0.1, 1, 10, 100, 1000],
}

stacking_grid = GridSearchCV(
    base_stacking,
    meta_params,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    verbose=1
)

stacking_grid.fit(X_train, y_train)
stacking_grid_pred = stacking_grid.predict(X_val)
stacking_grid_rmse = np.sqrt(mean_squared_error(y_val, stacking_grid_pred))
stacking_grid_r2 = r2_score(y_val, stacking_grid_pred)

print(f"OK - Best meta-learner alpha: {stacking_grid.best_params_['final_estimator__alpha']}")
print(f"OK - Best CV score: {-stacking_grid.best_score_:,.0f}")
print(f"OK - Validation RMSE: {stacking_grid_rmse:,.0f}")
print(f"OK - Validation R²: {stacking_grid_r2:.4f}")

# ============================================================================
# PHASE 7: COMPARISON WITH ALL MODELS INCLUDING STACKING
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON (INDIVIDUAL + STACKING)")
print("="*80)

# Update results with stacking models
results['Stacking_Ridge'] = {
    'rmse': stacking_ridge_rmse,
    'r2': stacking_ridge_r2,
    'model': stacking_ridge
}

results['Stacking_Lasso'] = {
    'rmse': stacking_lasso_rmse,
    'r2': stacking_lasso_r2,
    'model': stacking_lasso
}

results['Stacking_GB'] = {
    'rmse': stacking_gb_rmse,
    'r2': stacking_gb_r2,
    'model': stacking_gb
}

results['Stacking_GridSearch'] = {
    'rmse': stacking_grid_rmse,
    'r2': stacking_grid_r2,
    'model': stacking_grid.best_estimator_
}

# Create comprehensive results summary
comprehensive_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Validation RMSE': [results[model]['rmse'] for model in results.keys()],
    'Validation R²': [results[model]['r2'] for model in results.keys()],
})

comprehensive_df = comprehensive_df.sort_values('Validation RMSE').reset_index(drop=True)
comprehensive_df['Rank'] = range(1, len(comprehensive_df) + 1)

print("\n" + comprehensive_df.to_string(index=False))

# Identify best overall model
best_overall_model_name = comprehensive_df.iloc[0]['Model']
best_overall_rmse = comprehensive_df.iloc[0]['Validation RMSE']
best_overall_r2 = comprehensive_df.iloc[0]['Validation R²']

print(f"\n{'='*80}")
print(f"BEST OVERALL MODEL: {best_overall_model_name}")
print(f"{'='*80}")
print(f"Validation RMSE: ${best_overall_rmse:,.0f}")
print(f"Validation R²: {best_overall_r2:.4f}")
print(f"Improvement over baseline (Ridge): {(results['Ridge']['rmse'] - best_overall_rmse) / results['Ridge']['rmse'] * 100:.2f}%")

# Visualize comprehensive comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. All models RMSE
ax1 = axes[0, 0]
colors = ['red' if 'Stacking' in model else 'steelblue' for model in comprehensive_df['Model']]
ax1.barh(comprehensive_df['Model'], comprehensive_df['Validation RMSE'], color=colors)
ax1.set_xlabel('Validation RMSE ($)')
ax1.set_title('All Models - RMSE Comparison')
ax1.invert_yaxis()
for i, v in enumerate(comprehensive_df['Validation RMSE']):
    ax1.text(v, i, f' ${v:,.0f}', va='center')

# 2. All models R²
ax2 = axes[0, 1]
ax2.barh(comprehensive_df['Model'], comprehensive_df['Validation R²'], color=colors)
ax2.set_xlabel('Validation R²')
ax2.set_title('All Models - R² Comparison')
ax2.invert_yaxis()
for i, v in enumerate(comprehensive_df['Validation R²']):
    ax2.text(v, i, f' {v:.4f}', va='center')

# 3. Stacking models only comparison
ax3 = axes[1, 0]
stacking_only = comprehensive_df[comprehensive_df['Model'].str.contains('Stacking')]
ax3.bar(range(len(stacking_only)), stacking_only['Validation RMSE'], color='darkred', alpha=0.7)
ax3.set_xticks(range(len(stacking_only)))
ax3.set_xticklabels(stacking_only['Model'], rotation=45, ha='right')
ax3.set_ylabel('Validation RMSE ($)')
ax3.set_title('Stacking Models - RMSE Comparison')
ax3.grid(axis='y', alpha=0.3)

# 4. Stacking R² comparison
ax4 = axes[1, 1]
ax4.bar(range(len(stacking_only)), stacking_only['Validation R²'], color='darkgreen', alpha=0.7)
ax4.set_xticks(range(len(stacking_only)))
ax4.set_xticklabels(stacking_only['Model'], rotation=45, ha='right')
ax4.set_ylabel('Validation R²')
ax4.set_title('Stacking Models - R² Comparison')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('06_stacking_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PHASE 8: FEATURE IMPORTANCE FROM BASE LEARNERS
# ============================================================================

print("\n" + "="*80)
print("PHASE 8: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance from tree-based base learners
feature_importance_dict = {}

if hasattr(results['RandomForest']['model'], 'feature_importances_'):
    feature_importance_dict['RandomForest'] = results['RandomForest']['model'].feature_importances_

if hasattr(results['GradientBoosting']['model'], 'feature_importances_'):
    feature_importance_dict['GradientBoosting'] = results['GradientBoosting']['model'].feature_importances_

if hasattr(results['XGBoost']['model'], 'feature_importances_'):
    feature_importance_dict['XGBoost'] = results['XGBoost']['model'].feature_importances_

# Plot feature importance
if feature_importance_dict:
    fig, axes = plt.subplots(1, len(feature_importance_dict), figsize=(6*len(feature_importance_dict), 6))
    
    if len(feature_importance_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, importances) in enumerate(feature_importance_dict.items()):
        # Get top 15 features
        feature_imp_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        axes[idx].barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='teal')
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{model_name} - Top 15 Features')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('07_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\nOK - Stacking and feature importance analysis completed!")
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total models trained: {len(results)}")
print(f"Base learners in stacking: {len(base_learners)}")
print(f"Meta-learners tested: 4 (Ridge, Lasso, GradientBoosting, GridSearch)")
print(f"Best model: {best_overall_model_name}")
print(f"Best RMSE: ${best_overall_rmse:,.0f}")
print(f"Best R²: {best_overall_r2:.4f}")

# ============================================================================
# PHASE 9: TRAIN BEST MODEL ON FULL DATA AND CREATE KAGGLE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("PHASE 9: TRAIN BEST MODEL ON FULL DATA AND CREATE KAGGLE SUBMISSION")
print("="*80)

# Decide which model and features to use
use_pretrained = xgb_pretrained_loaded

if use_pretrained:
    print(f"Using pre-trained XGBoost model with enhanced features...")
    # Use the pre-trained model directly (no need to refit)
    preds_test = xgb_best_model.predict(X_full_test)
    submission_df = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': preds_test
    })
else:
    print(f"Preparing submission with {best_overall_model_name} (simple numeric features)...")
    
    # Recreate the numeric feature set used earlier
    numeric_cols_full = train.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numeric_cols_full:
        numeric_cols_full.remove('Id')
    if 'SalePrice' in numeric_cols_full:
        numeric_cols_full.remove('SalePrice')

    # Fill missing values in train and test using train medians
    train_final = train[numeric_cols_full].fillna(train[numeric_cols_full].median())
    y_final = train['SalePrice']

    # Apply same outlier removal as during training
    if 'GrLivArea' in train_final.columns:
        outlier_idx_final = (train_final['GrLivArea'] > 4000) & (y_final < 300000)
        if outlier_idx_final.any():
            train_final = train_final[~outlier_idx_final]
            y_final = y_final[~outlier_idx_final]

    print(f"Full training set for final fit: {train_final.shape}")

    # Select the best model object from results and fit on full training data
    if best_overall_model_name in results:
        final_model = results[best_overall_model_name]['model']
    else:
        final_model = results.get('GradientBoosting', {}).get('model')

    if final_model is None:
        print("ERROR - No final model available to fit. Submission not created.")
    else:
        print(f"Fitting final model: {best_overall_model_name} on full training data...")
        final_model.fit(train_final, y_final)

        # Prepare test features (use same numeric columns and medians)
        test_features = test[numeric_cols_full].fillna(train[numeric_cols_full].median())

        # Predict and save submission
        preds_test = final_model.predict(test_features)
        submission_df = pd.DataFrame({
            'Id': test['Id'],
            'SalePrice': preds_test
        })

# Save submission
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"OK - Submission saved to {submission_path} ({len(submission_df)} rows)")
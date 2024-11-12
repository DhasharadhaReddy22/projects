import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_z_score(df, features=None, threshold=3):
    """Detect unique outlier indices in specified columns of a DataFrame using Z-scores.

    This function identifies data points that are considered outliers based on the Z-score,
    which measures how many standard deviations a data point is from the mean. Outliers are
    those data points with a Z-score greater than the specified threshold.
    """
    if features is None:
        features = df.columns

    outlier_set = set() 
    for feature in features:
        mean = df[feature].mean()
        std_dev = df[feature].std()
        z_scores = abs(df[feature] - mean) / std_dev
        
        outlier_indices = df[z_scores > threshold].index
        outlier_set.update(outlier_indices)
        
    print("Total Outliers:", len(outlier_set))
    
    return sorted(outlier_set)

def detect_outliers_iqr(df, features=None):
    """Detect unique outlier indices in specified columns of a DataFrame using IQR.

    This function identifies data points that are considered outliers based on the Interquartile Range (IQR).
    Outliers are those data points that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.
    """
    if features is None:
        features = df.columns

    outlier_set = set()
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1 
        
        lower_bound = Q1 - (IQR * 1.5)
        upper_bound = Q3 + (IQR * 1.5)
        
        outlier_indices = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index
        outlier_set.update(outlier_indices)
        
    print("Total Outliers:", len(outlier_set))
    return sorted(outlier_set)

def detect_outliers_lof(df, features=None, n_neighbors=20):
    """Detect unique outlier indices in specified columns of a DataFrame using Local Outlier Factor (LOF).

    This function identifies data points that are considered outliers based on the Local Outlier Factor (LOF).
    LOF measures the local deviation of density of a data point with respect to its neighbors.
    """
    if features is None:
        features = df.columns

    data = df[features].values
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    
    # Fit the model and predict outliers (-1 indicates an outlier)
    outliers = lof.fit_predict(data)
    
    outlier_indices = np.where(outliers == -1)[0]
    
    outlier_set = set(outlier_indices)
    
    print("Total Outliers:", len(outlier_set))
    return sorted(outlier_set)

def detect_outliers_isolation_forest(df, features=None, n_estimators=100):
    """
    Detect unique outlier indices in specified columns of a DataFrame using Isolation Forest.

    This function identifies data points that are considered outliers based on the Isolation Forest algorithm.
    Isolation Forest works by isolating observations using random decision trees and scoring their isolation level.
    """
    if features is None:
        features = df.columns

    data = df[features].values
    
    iso_forest = IsolationForest(n_estimators=n_estimators)
    
    # Fit the model and predict outliers (-1 indicates an outlier)
    outliers = iso_forest.fit_predict(data)
    
    # Find the indices of the outliers
    outlier_indices = np.where(outliers == -1)[0]
    
    outlier_set = set(outlier_indices)
    
    print("Total Outliers:", len(outlier_set))
    return sorted(outlier_set)

def detect_outliers_dbscan(df, features=None, epsilon=3, min_samples=20):
    """
    Detect outliers using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    """
    if features is None:
        features = df.columns
        
    data = df[features].values
    
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit the model and predict outliers (-1 indicates an outlier)
    labels = dbscan.fit_predict(data)
    
    outlier_indices = np.where(labels == -1)[0]
    
    outlier_set = set(outlier_indices)
    
    print("Total Outliers:", len(outlier_set))
    return sorted(outlier_set)
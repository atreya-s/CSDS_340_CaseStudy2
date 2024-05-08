# TODO for write up talk about using the elbow method to find k but that had to be automated now
# TODO say pca did not return good results we tried pca and pca to find the number of features then doing sfs
import functools
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, KMeans, AgglomerativeClustering, Birch, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score


seed = 0
show_elbow = False

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def predictor_baseline(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # k-means with K = number of unique VIDs of set1
    K = 20
    model = KMeans(n_clusters=K, random_state=seed, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv', 'set3.csv'] # TODO remove the set3 here
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')


def evaluate_path(csv_path):
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of {csv_path}: {rand_index_score:.4f}')

def pca_n_components(X, pca_threshold, seed=None):
    pca = PCA(random_state=seed)
    pca.fit(X)
    total_variance = sum(pca.explained_variance_)
    explained_variance_ratio = pca.explained_variance_ / total_variance

    num_components = 0
    cur_v = 0
    for v in sorted(explained_variance_ratio, reverse=True):
        num_components += 1
        cur_v += v
        if cur_v > pca_threshold:
            break
    return num_components

def find_elbow_point(distortions):
    # list of numbers from this distortion to next distortion
    jumps = [(distortions[i] - distortions[i+1]) for i in range(len(distortions)-1)]

    max_jump = max(jumps)
    for i in range(1, len(jumps) - 2):
        ret_from_prev = jumps[i] < (1 / 100) * jumps[i - 1]
        ret_from_max = jumps[i] < (1 / 100) * max_jump
        if  ret_from_prev or ret_from_max:
            return i - 1

    return len(jumps) - 1

def sfs_preprocessor(X, k, n_feat):
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=k, random_state=seed)
    if n_feat == X.shape[1]:
        return Pipeline([
            ('scaler', scaler)
            ])
    sfs = SequentialFeatureSelector(kmeans, direction='forward',
                                    n_features_to_select=n_feat)
    return Pipeline([
        ('scaler', scaler),
        ('feature_selection', sfs)
    ])

def pca_preprocessor(target_explained_variance, X):
    scaler = StandardScaler()
    n_feat = pca_n_components(scaler.fit_transform(X), target_explained_variance)
    pca = PCA(n_components=n_feat)
    return Pipeline([
        ('scaler', scaler),
        ('pca', pca)
        ])

def scaler_preprocessor():
    scaler = StandardScaler()
    return Pipeline([
        ('scaler', scaler),
        ])


def predictor(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()

    distortions = []
    min_clusters = 2
    max_clusters = 30
    preprocessor = scaler_preprocessor()
    x_pre = preprocessor.fit_transform(X)
    # for sfs feature selection find the number of features given target variance
    # n_feat = pca_n_components(X, pca_threshold=0.85)
    for k in range(min_clusters, max_clusters + 1):
        # for sfs selection need to change the rates to be for this k clusters
        # preprocessor = sfs_preprocessor(X, k, n_feat)
        # x_pre = preprocessor.fit_transform(X)
        clusterer = KMeans(n_clusters=k, random_state=seed)
        clusterer.fit(x_pre)
        distortions.append(clusterer.inertia_)

    if show_elbow:
        graph(distortions, max_clusters)

    k = find_elbow_point(distortions)
    print(f"n_clusters found: {k}")
    # preprocessor = sfs_preprocessor(X, k, n_feat)
    preprocessor = scaler_preprocessor()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GaussianMixture(n_components=k, random_state=seed))
    ])

    labels_pred = model.fit_predict(X)

    return labels_pred


def graph(distortions, max_clusters):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), distortions, marker='o', linestyle='-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Distortion vs Number of Clusters')
    plt.xticks(range(2, max_clusters + 1))
    plt.grid(True)
    plt.show()


def print_num_clusters(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    num_clusters = df['VID'].nunique()
    print(f"n_clusters for {csv_path}: {num_clusters}")


if __name__=="__main__":
    get_baseline_score()
    for f in ['set1.csv', 'set2.csv', 'set3.csv']:
        print_num_clusters('./Data/' + f)
        evaluate_path('./Data/' + f)
    # evaluate()

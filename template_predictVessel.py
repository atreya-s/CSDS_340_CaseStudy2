import pandas as pd
import numpy as np
from sklearn import preprocessing
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import VotingClassifier

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
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
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


def predictor(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})

    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3)
    gmm = GaussianMixture(n_components=3)
    hierarchical = AgglomerativeClustering(n_clusters=3)

    kmeans_labels = kmeans.fit_predict(X_scaled)
    gmm_labels = gmm.fit_predict(X_scaled)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    ensemble_labels = (kmeans_labels + gmm_labels + hierarchical_labels) // 3

    return ensemble_labels


def evaluate_path(csv_path):
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of {csv_path}: {rand_index_score:.4f}')


if __name__=="__main__":
    get_baseline_score()
    for f in ['set1.csv', 'set2.csv']:
        evaluate_path('./Data/' + f)
    # evaluate()

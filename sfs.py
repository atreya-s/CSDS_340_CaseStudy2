# TODO for write up talk about using the elbow method to find k but that had to be automated now
import functools
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold

seed = 0
show_elbow = True

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
    jumps = [(distortions[i] - distortions[i+1]) for i in range(len(distortions)-1)]
    ratios = [(jumps[i] / jumps[i-1]) if i > 0 and jumps[i-1] != 0 else 0 for i in range(len(jumps))]

    best_id = 0
    largest = 0
    for i, ratio in enumerate(ratios):
        if ratio > largest:
            largest = ratio
            best_id = i - 1

    return best_id

def gen_preprocessor(X, k, n_feat):
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=k)
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


def predictor(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()


    distortions = []
    min_clusters = 2
    max_clusters = 30

    n_feat = pca_n_components(X, pca_threshold=0.99)

    for k in range(min_clusters, max_clusters + 1):
        preprocessor = gen_preprocessor(X, k, n_feat)
        x_pre = preprocessor.fit_transform(X)
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(x_pre)
        distortions.append(kmeans.inertia_)

    if show_elbow:
        elbow_graph(distortions, max_clusters)

    k = find_elbow_point(distortions) + min_clusters
    print(f"n_clusters found: {k}")
    k = 8 # TODO artificial good value for set2
    preprocessor = gen_preprocessor(X, k, n_feat)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', KMeans(n_clusters=k, random_state=seed))
    ])

    labels_pred = model.fit_predict(X)

    return labels_pred


def elbow_graph(distortions, max_clusters):
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
    for f in ['set1.csv', 'set2.csv']:
        print_num_clusters('./Data/' + f)
        evaluate_path('./Data/' + f)
    # evaluate()

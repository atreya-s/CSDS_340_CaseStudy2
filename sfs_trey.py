import functools
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

seed = 0
show_elbow = True

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc * 60 + x, map(int, hh_mm_ss.split(':')))

def load_data(csv_path):
    return pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})

def standardize_data(X):
    return preprocessing.StandardScaler().fit_transform(X)

def pca_n_components(X, threshold=0.95):
    pca = PCA(n_components=threshold, svd_solver='full', random_state=seed)
    pca.fit(X)
    return pca.n_components_

def determine_k(X):
    distortions = []
    K_range = range(2, 31)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=seed)
        model.fit(X)
        distortions.append(model.inertia_)

    if show_elbow:
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, distortions, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    # Improved elbow detection using second derivative concept [WCSS]
    kinks = [distortions[i - 1] - 2 * distortions[i] + distortions[i + 1] for i in range(1, len(distortions) - 1)]
    k = kinks.index(max(kinks)) + 2  # Plus 2 because range starts from 2
    return k

def predictor(csv_path):
    df = load_data(csv_path)
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    X = standardize_data(X)

    n_components = pca_n_components(X)
    pca = PCA(n_components=n_components, random_state=seed)
    X = pca.fit_transform(X)

    k = determine_k(X)
    model = KMeans(n_clusters=k, random_state=seed)
    labels_pred = model.fit_predict(X)
    return labels_pred

def evaluate_clusters(csv_path):
    df = load_data(csv_path)
    labels_true = df['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    return adjusted_rand_score(labels_true, labels_pred)

if __name__ == "__main__":
    file_paths = ['./Data/set1.csv', './Data/set2.csv']
    for path in file_paths:
        score = evaluate_clusters(path)
        print(f'Adjusted Rand Index Score of {path}: {score:.4f}')


import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster(df, max_clusters=25):
    X = df[['avg_yearly_returns', 'variance']].values
    rb = RobustScaler()
    X_rb = rb.fit_transform(X)

    sse_within_cluster = {}
    silhouette_scores = {}

    for k in range(2, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
        kmeans.fit(X_rb)
        sse_within_cluster[k] = kmeans.inertia_

        if len(set(kmeans.labels_)) > 1 and len(X_rb) > k:
            silhouette_scores[k] = metrics.silhouette_score(X_rb, kmeans.labels_)
        else:
            silhouette_scores[k] = float('nan')

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(list(sse_within_cluster.keys()), list(sse_within_cluster.values()))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE Within Cluster")
    plt.title("Within Cluster SSE After K-Means Clustering")

    plt.subplot(212)
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score After K-Means Clustering")
    plt.show()

def apply_kmeans(df, clusters):
    rb = RobustScaler()
    X_rb = rb.fit_transform(df[['avg_yearly_returns', 'variance']])
    kmeans = KMeans(n_clusters=clusters, random_state=10, n_init=10)
    df.loc[:, 'cluster'] = kmeans.fit_predict(X_rb)  # Use .loc[] to avoid the warning
    return df


def trail_one():
    # Load the processed data
    agg_df3 = pd.read_csv('./processed_data.csv')

    # Handle NaN values by dropping them
    agg_df3_clean = agg_df3.dropna()

    print("Starting Trail One: Full Dataset Clustering")
    
    # Plot clusters and apply KMeans
    plot_cluster(agg_df3_clean)
    clustered_df = apply_kmeans(agg_df3_clean, clusters=25)
    
    # Save the clustered data to a CSV file
    clustered_df.to_csv('./trail_one_output.csv', index=False)

    return clustered_df

def trail_two(clustered_df):
    sub_cluster_df = clustered_df[clustered_df['cluster'] == clustered_df['cluster'].mode()[0]]
    n_samples = len(sub_cluster_df)

    # Adjust the number of clusters based on the number of samples
    clusters = min(25, n_samples - 1) if n_samples > 1 else 1

    print(f"Starting Trail Two: Sub-cluster Clustering with {clusters} clusters and {n_samples} samples.")
    
    if n_samples > 1:  # Only plot and apply KMeans if there is more than 1 sample
        plot_cluster(sub_cluster_df, max_clusters=clusters)
        sub_clustered_df = apply_kmeans(sub_cluster_df, clusters=clusters)
        sub_clustered_df.to_csv('./trail_two_output.csv', index=False)
    else:
        print("Not enough samples to apply KMeans clustering.")

    return sub_cluster_df if n_samples > 1 else clustered_df

def trail_three(sub_clustered_df):
    best_sub_cluster = sub_clustered_df[sub_clustered_df['cluster'] == sub_clustered_df['cluster'].mode()[0]]
    n_samples = len(best_sub_cluster)

    # Adjust the number of clusters based on the number of samples
    clusters = min(15, n_samples - 1) if n_samples > 1 else 1

    print(f"Starting Trail Three: Best Sub-cluster Clustering with {clusters} clusters and {n_samples} samples.")
    
    if n_samples > 1:  # Only plot and apply KMeans if there are more than 1 sample
        plot_cluster(best_sub_cluster, max_clusters=clusters)
        best_clustered_df = apply_kmeans(best_sub_cluster, clusters=clusters)
        best_clustered_df.to_csv('./trail_three_output.csv', index=False)
    else:
        print("Not enough samples to apply KMeans clustering.")
        best_clustered_df = best_sub_cluster  # Return the sub-clustered dataframe without any changes

    return best_clustered_df


if __name__ == "__main__":
    clustered_df = trail_one()
    sub_clustered_df = trail_two(clustered_df)
    final_clustered_df = trail_three(sub_clustered_df)
    print("Final Clustering Results Saved.") 

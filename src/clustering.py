from building import load_anguillara
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

building_list = load_anguillara()

# Obtain an unique dataframe appending all the load profiles for each consumer

df_dict = {}

for building in building_list:
    if building.building_info["user_type"] == "consumer":
        data = building.energy_meter.data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data.set_index("timestamp", inplace=True)
        data = data.resample("H").mean()
        data_pivot = data.pivot_table(index=data.index.date, columns=data.index.strftime("%H:%M"), values="Net", dropna=True)
        # Normalize the data using Z-score
        data_pivot = (data_pivot - data_pivot.mean()) / data_pivot.std()
        data_pivot["building"] = building.building_info["name"]
        # Divide into weekdays and weekends
        data_pivot.index = pd.to_datetime(data_pivot.index)
        data_pivot["weekday"] = data_pivot.index.weekday
        data_pivot["day_type"] = "Weekday"
        data_pivot.loc[data_pivot["weekday"] >= 5, "day_type"] = "Weekend"
        data_pivot.drop(columns=["weekday"], inplace=True)
        df_dict[building.building_info["name"]] = data_pivot

df = pd.concat(df_dict.values())

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(df[df.columns[:23]])
df["cluster"] = kmeans.labels_

# Plot the load profiles for each cluster

fig, ax = plt.subplots(figsize=(12, 5), ncols=5, nrows=2)
hour_labels = df.columns[:23:4]

for i in range(10):
    axes = ax[i // 5, i % 5]
    cluster = df[df["cluster"] == i].drop(columns=["cluster", "building", "day_type"])
    centroid = cluster.mean()
    # Take also the first and third quartile
    q1 = cluster.quantile(0.25)
    q3 = cluster.quantile(0.75)
    for idx, row in cluster.iterrows():
        load_profile_line, = axes.plot(row.index, row.values, color="grey", alpha=0.2)
    centroid_line, = axes.plot(centroid.index, centroid.values, color="red", label="Centroid")
    q1_line = axes.plot(q1.index, q1.values, color="blue", linestyle="--", label="Q1")
    q3_line = axes.plot(q3.index, q3.values, color="green", linestyle="--", label="Q3")
    axes.set_title(f"Cluster {i}")
    axes.set_xlabel("Hour of the day", fontsize=12)
    axes.set_ylabel("Power (kW)", fontsize=12)
    plt.xticks()
    axes.set_xticks(range(0, 23, 4))

# Extract the legend from the last subplot
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=12, ncol=5, fancybox=True, shadow=True)
plt.tight_layout(rect=(0, 0.08, 1, 0.98))
plt.show()

# Calculate the percentage of load profiles as weekday and weekend in each cluster
day_type_count = df.groupby(["cluster", "day_type"]).size().unstack().fillna(0).apply(lambda x: x / x.sum(), axis=1)
# Stacked bar plot of the day_type_count
day_type_count.plot(kind="bar", stacked=True)
plt.show()


cluster_counts = df["cluster"].value_counts(normalize=True).sort_index()

# df = df.groupby("cluster").filter(lambda x: len(x) / len(df) > 0.05)
building_cluster = df.groupby("building")["cluster"].value_counts(dropna=False).unstack(fill_value=0)

silhouette_score = silhouette_score(df[df.columns[:23]], kmeans.labels_)
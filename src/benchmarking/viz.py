import matplotlib.pyplot as plt
import pandas as pd
import os
from settings import PROJECT_ROOT
from src.building import Building, load_anguillara, load_garda
import seaborn as sns


def plot_silhouette_scores(scores: pd.DataFrame):
    """
    Grafica il silhouette score per un numero di cluster variabile
    :param scores: dataframe con silhouette score per un numero di cluster variabile
    """

    max_score = scores["silhouette_score"].max()
    max_cluster = scores[scores["silhouette_score"] == max_score]["n_clusters"].values[0]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(scores["n_clusters"], scores["silhouette_score"])
    plt.scatter(max_cluster, max_score, color='red')
    plt.hlines(max_score, scores["n_clusters"].min(), scores["n_clusters"].max(), linestyles='dashed', colors='red')
    plt.vlines(max_cluster, scores["silhouette_score"].min(), scores["silhouette_score"].max(), linestyles='dashed',
               colors='red')
    plt.xlabel("Numero di cluster")
    plt.ylabel("Silhouette score")
    plt.xticks(scores["n_clusters"])
    plt.title("Silhouette score al variare del numero di cluster")

    return fig


def plot_load_profiles_user(user_id: str, aggregate: str):
    """
    Grafica i profili di carico di un utente dividendoli per cluster
    :param user_id: id dell'utente
    :param aggregate: nome dell'aggregato ("anguillara" o "garda")
    """

    building = Building(user_id, aggregate)
    data = building.energy_meter.data

    cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}.csv"))
    cluster['cluster'] = cluster.apply(lambda row: 'C' + str(row['cluster']) if row['user_type'] == 'consumer' else 'P' + str(row['cluster']), axis=1)
    cluster_user = cluster[cluster["user_id"] == user_id]
    cluster_user["date"] = pd.to_datetime(cluster_user["date"]).dt.date

    # Merge data with cluster user on the date
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date

    data_cluster = pd.merge(data[["timestamp", "date", "Load"]], cluster_user[["date", "cluster"]], on="date", how="inner")
    data_cluster["hour"] = data_cluster["timestamp"].dt.strftime("%H:%M")

    cluster_labels = cluster.loc[cluster["user_type"] == building.building_info["user_type"], "cluster"].unique()
    cluster_labels.sort()

    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels.size, sharey=True)

    palette = sns.color_palette("colorblind", cluster_labels.size)

    for i, cluster_label in enumerate(cluster_labels):
        data_cluster_viz = data_cluster[data_cluster["cluster"] == cluster_label]
        data_cluster_viz_grouped = data_cluster_viz.groupby("date")

        for date, group in data_cluster_viz_grouped:
            ax[i].plot(group["hour"], group["Load"], color=palette[i], alpha=0.1, linewidth=0.3)

        ax[i].set_title(f"Cluster {cluster_label}")
        ax[i].set_xlabel("Ora del giorno")
        ax[i].set_ylabel("Potenza [kW]")
        ax[i].set_xticks(range(0, 96, 16))
        ax[i].tick_params(axis='x', labelsize=9)

    centroid_load_profiles = data_cluster[["hour", "Load", "cluster"]].groupby(["hour", "cluster"]).median()
    centroid_load_profiles.reset_index(inplace=True)
    q1 = data_cluster[["hour", "Load", "cluster"]].groupby(["hour", "cluster"]).quantile(0.25)
    q1.reset_index(inplace=True)
    q3 = data_cluster[["hour", "Load", "cluster"]].groupby(["hour", "cluster"]).quantile(0.75)
    q3.reset_index(inplace=True)

    for j, cluster_label in enumerate(cluster_labels):
        cluster_data = centroid_load_profiles[centroid_load_profiles["cluster"] == cluster_label]
        cluster_q1 = q1[q1["cluster"] == cluster_label]
        cluster_q3 = q3[q3["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load"], color=color)
        ax[j].fill_between(cluster_q1["hour"], cluster_q1.set_index("hour")["Load"],
                           cluster_q3.set_index("hour")["Load"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.01, 1, 0.92), h_pad=4)
    plt.ylim(0, building.building_info["rated_power"] * 1000)
    plt.suptitle(f"Profili di carico divisi per cluster per l'utente {building.building_info['name']}",
                 fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"load_profiles_{user_id}.png"))


def plot_load_profiles_aggregate(aggregate: str):
    """
    Grafica i profili di carico di un aggregato dividendoli per cluster
    :param aggregate: nome dell'aggregato ("anguillara" o "garda")
    """

    cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}.csv"))
    cluster['cluster'] = cluster.apply(lambda row: 'C' + str(row['cluster']) if row['user_type'] == 'consumer' else 'P' + str(row['cluster']), axis=1)
    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date

    cluster_consumer = cluster[cluster["user_type"] == "consumer"]
    cluster_labels_consumer = cluster_consumer["cluster"].unique()
    cluster_labels_consumer.sort()

    cluster_prosumer = cluster[cluster["user_type"] != "consumer"]
    cluster_labels_prosumer = cluster_prosumer["cluster"].unique()
    cluster_labels_prosumer.sort()

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    building_data_list = []
    for building in building_list:
        data = building.energy_meter.data
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["hour"] = data["timestamp"].dt.strftime("%H:%M")
        data["date"] = data["timestamp"].dt.date
        data["building_name"] = building.building_info["name"]
        data["Load_norm"] = data["Load"] / data.groupby("date")["Load"].transform("max")

        cluster_user = cluster[cluster["building_name"] == building.building_info["name"]]
        data = pd.merge(data, cluster_user, on="date", how="inner")
        building_data_list.append(data)

    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels_consumer.size, sharey=True)
    palette = sns.color_palette("colorblind", cluster_labels_consumer.size)

    for building in building_data_list:
        for i, cluster_label in enumerate(cluster_labels_consumer):
            data_cluster = building[building["cluster"] == cluster_label]
            data_cluster_grouped = data_cluster.groupby("date")

            for date, group in data_cluster_grouped:
                ax[i].plot(group["hour"], group["Load_norm"], color=palette[i], alpha=0.1, linewidth=0.2)
            ax[i].set_title(f"Cluster {cluster_label}")
            ax[i].set_xlabel("Ora del giorno")
            ax[i].set_ylabel("Potenza normalizzata [-]")
            ax[i].set_xticks(range(0, 96, 16))
            ax[i].tick_params(axis='x', labelsize=9)

    total_load_profiles = pd.concat(building_data_list)
    centroid_load_profiles = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(["hour", "cluster"]).median()
    centroid_load_profiles.reset_index(inplace=True)
    q1 = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.25)
    q1.reset_index(inplace=True)
    q3 = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.75)
    q3.reset_index(inplace=True)

    for j, cluster_label in enumerate(cluster_labels_consumer):
        cluster_data = centroid_load_profiles[centroid_load_profiles["cluster"] == cluster_label]
        cluster_q1 = q1[q1["cluster"] == cluster_label]
        cluster_q3 = q3[q3["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load_norm"], color=color, label="Centroid Consumer")
        ax[j].fill_between(cluster_q1["hour"], cluster_q1.set_index("hour")["Load_norm"],
                           cluster_q3.set_index("hour")["Load_norm"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico dei CONSUMER nei cluster per {aggregate.title()}", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"load_profiles_{aggregate}_consumer.png"))

    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels_prosumer.size, sharey=True)
    palette = sns.color_palette("colorblind", cluster_labels_prosumer.size)

    for building in building_data_list:
        for i, cluster_label in enumerate(cluster_labels_prosumer):
            data_cluster = building[building["cluster"] == cluster_label]
            data_cluster_grouped = data_cluster.groupby("date")

            for date, group in data_cluster_grouped:
                ax[i].plot(group["hour"], group["Load_norm"], color=palette[i], alpha=0.1, linewidth=0.2)
            ax[i].set_title(f"Cluster {cluster_label}")
            ax[i].set_xlabel("Ora del giorno")
            ax[i].set_ylabel("Potenza normalizzata [-]")
            ax[i].set_xticks(range(0, 96, 16))
            ax[i].tick_params(axis='x', labelsize=9)

    total_load_profiles = pd.concat(building_data_list)
    centroid_load_profiles = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(["hour", "cluster"]).median()
    centroid_load_profiles.reset_index(inplace=True)
    q1 = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.25)
    q1.reset_index(inplace=True)
    q3 = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.75)
    q3.reset_index(inplace=True)

    for j, cluster_label in enumerate(cluster_labels_prosumer):
        cluster_data = centroid_load_profiles[centroid_load_profiles["cluster"] == cluster_label]
        cluster_q1 = q1[q1["cluster"] == cluster_label]
        cluster_q3 = q3[q3["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load_norm"], color=color, label="Centroid Prosumer")
        ax[j].fill_between(cluster_q1["hour"], cluster_q1.set_index("hour")["Load_norm"],
                           cluster_q3.set_index("hour")["Load_norm"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico dei PROSUMER nei cluster per {aggregate.title()}", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"load_profiles_{aggregate}_prosumer.png"))


def plot_cluster_percentage(aggregate: str):
    """
    Grafica la percentuale di profili in ogni cluster per ogni utente dell'aggregato
    :param aggregate: nome dell'aggregato ("anguillara" o "garda")
    """

    cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}.csv"))
    cluster['cluster'] = cluster.apply(lambda row: 'C' + str(row['cluster']) if row['user_type'] == 'consumer' else 'P' + str(row['cluster']), axis=1)

    consumer_cluster = cluster[cluster["user_type"] == "consumer"]
    consumer_cluster_grouped = consumer_cluster.groupby(["building_name", "cluster"]).size().reset_index(name="count")
    consumer_cluster_grouped["percentage"] = consumer_cluster_grouped.groupby("building_name")["count"].transform(
        lambda x: x / x.sum() * 100)

    df_cluster_percentage = consumer_cluster_grouped.pivot(index="building_name", columns="cluster", values="percentage")
    df_cluster_percentage.fillna(0, inplace=True)
    df_cluster_percentage = df_cluster_percentage.round(1)

    palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
    ax.set_xlabel("Percentuale [%]", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)
    ax.set_title(f"Percentuale di profili di carico dei CONSUMER nei cluster per {aggregate.title()}",
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    for container in bars.containers:
        for rect in container:
            width = rect.get_width()
            if width > 0:
                x = rect.get_x() + width / 2
                y = rect.get_y() + rect.get_height() / 2
                percentage = f"{width:.1f}%"
                ax.text(x, y, percentage, ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_cluster_percentage.columns),
              fancybox=True, shadow=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"cluster_percentage_{aggregate}_consumer.png"))

    producer_cluster = cluster[cluster["user_type"] != "consumer"]
    producer_cluster_grouped = producer_cluster.groupby(["building_name", "cluster"]).size().reset_index(name="count")
    producer_cluster_grouped["percentage"] = producer_cluster_grouped.groupby("building_name")["count"].transform(
        lambda x: x / x.sum() * 100)

    df_cluster_percentage = producer_cluster_grouped.pivot(index="building_name", columns="cluster", values="percentage")
    df_cluster_percentage.fillna(0, inplace=True)
    df_cluster_percentage = df_cluster_percentage.round(1)

    palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
    ax.set_xlabel("Percentuale [%]", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)
    ax.set_title(f"Percentuale di profili di carico dei PROSUMER nei cluster per {aggregate.title()}",
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    for container in bars.containers:
        for rect in container:
            width = rect.get_width()
            if width > 0:
                x = rect.get_x() + width / 2
                y = rect.get_y() + rect.get_height() / 2
                percentage = f"{width:.1f}%"
                ax.text(x, y, percentage, ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_cluster_percentage.columns),
                fancybox=True, shadow=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"cluster_percentage_{aggregate}_prosumer.png"))

import datetime

import pandas as pd
import numpy as np
import os
from settings import PROJECT_ROOT
from building import load_anguillara, load_garda
import shape_factor
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
global building_list

aggregate = "anguillara"

if aggregate == "anguillara":
    building_list = load_anguillara()
elif aggregate == "garda":
    building_list = load_garda()

cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}.csv"))

building_sf_dict = {}
for building in building_list:
    data = building.energy_meter.data.copy()
    data.set_index("timestamp", inplace=True)
    data = data["Net"].clip(lower=0)
    data = data.resample("1H").mean()
    mean_max = shape_factor.sf_mean_max(data)
    min_max = shape_factor.sf_min_max(data)
    min_mean = shape_factor.sf_min_mean(data)
    daytime_mean = shape_factor.sf_daytime_mean(data)
    daytime_max = shape_factor.sf_daytime_max(data)
    daytime_min_mean = shape_factor.sf_daytime_min_mean(data)
    nightime_mean = shape_factor.sf_nightime_mean(data)
    nightime_max = shape_factor.sf_nightime_max(data)
    nightime_min_mean = shape_factor.sf_nightime_min_mean(data)
    afternoon_mean = shape_factor.sf_afternoon_mean(data)
    afternoon_max = shape_factor.sf_afternoon_max(data)
    afternoon_min_mean = shape_factor.sf_afternoon_min_mean(data)
    peak_load = shape_factor.peakload(data)
    peak_period = shape_factor.peak_period(data, peak_load)

    df_sf = pd.concat([mean_max, min_max, min_mean, daytime_mean, daytime_max, daytime_min_mean, nightime_mean,
                       nightime_max, nightime_min_mean, afternoon_mean, afternoon_max, afternoon_min_mean,
                       peak_period], axis=1)
    df_sf.columns = ["mean_max", "min_max", "min_mean", "daytime_mean", "daytime_max", "daytime_min_mean",
                     "nightime_mean", "nightime_max", "nightime_min_mean", "afternoon_mean", "afternoon_max",
                     "afternoon_min_mean", "peak_night", "pick_morning", "peak_mid-day", "peak_evening"]
    df_sf.dropna(inplace=True)
    df_sf["building_name"] = building.building_info["name"]
    df_sf["user_type"] = building.building_info["user_type"]
    building_sf_dict[building.building_info["name"]] = df_sf

df_sf_total = pd.concat(building_sf_dict.values())
df_sf_total.reset_index(inplace=True, names=["date"])

cluster['date'] = pd.to_datetime(cluster['date'])
cluster["date"] = cluster["date"].dt.date

df_sf_total = df_sf_total.merge(cluster.drop(columns=["user_type"]), on=["date", "building_name"], how="left")
df_sf_total.set_index("date", inplace=True)

"""
Consumer analysis
"""

cluster_labels = np.sort(cluster[cluster["user_type"] == "consumer"]["cluster"].unique())

# Obtain load profiles for each building
building_data_list = []
df_cluster_percentage = pd.DataFrame(columns=cluster_labels,
                                     index=[building.building_info["name"] for building in building_list if building.building_info["user_type"] == "consumer"])
df_cluster_count = pd.DataFrame(columns=cluster_labels,
                                index=[building.building_info["name"] for building in building_list if building.building_info["user_type"] == "consumer"])
for building in building_list:
    if building.building_info["user_type"] == "consumer":
        data = building.energy_meter.data.copy()
        data.set_index("timestamp", inplace=True)
        data = data.resample("1H").mean()
        data["building_name"] = building.building_info["name"]
        data["date"] = data.index.date
        data["hour"] = data.index.strftime("%H:%M")
        data.reset_index(inplace=True)
        data["Net_norm"] = data["Net"] / data.groupby("date")["Net"].transform("max")

        data = data.merge(cluster, on=["building_name", "date"], how="left")
        data["user_type"] = building.building_info["user_type"]
        building_data_list.append(data)

        building_cluster = cluster[cluster["building_name"] == building.building_info["name"]]
        cluster_counts = building_cluster["cluster"].value_counts().reindex(cluster_labels).fillna(0)
        df_cluster_count.loc[building.building_info["name"]] = cluster_counts
        cluster_percentage = (cluster_counts / cluster_counts.sum() * 100).round(1)
        df_cluster_percentage.loc[building.building_info["name"]] = cluster_percentage

df_cluster_percentage = df_cluster_percentage.astype(float)
df_cluster_count = df_cluster_count.astype(int)

palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
fig, ax = plt.subplots(figsize=(10, 8))
bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
ax.set_xlabel("Percentuale [%]")
ax.set_ylabel("Edificio")
ax.set_title(f"Percentuale di profili di carico dei CONSUMER nei cluster per {aggregate.title()}")
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
plt.show()

palette = sns.color_palette("RdBu", n_colors=len(df_cluster_count.index))
fig, ax = plt.subplots(figsize=(10, 8))
bars = df_cluster_count.T.plot(kind="bar", stacked=True, ax=ax, color=palette)
ax.set_ylabel("Conteggio")
ax.set_xlabel("Cluster")
ax.set_title(f"Conteggio di profili di carico dei CONSUMER nei cluster per {aggregate.title()}")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_cluster_count.index),
            fancybox=True, shadow=True)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()

fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=len(cluster_labels))
unique_labels = []

for i, building in enumerate(building_data_list):

    load_profiles = building[["date", "hour", "Net_norm"]].pivot_table(index="date", columns="hour", values="Net_norm")
    cluster_dates = cluster[cluster["building_name"] == building_list[i].building_info["name"]]
    load_profiles = load_profiles.merge(cluster_dates, on="date", how="left")
    building_name = building_list[i].building_info["name"]

    for j, cluster_label in enumerate(cluster_labels):
        cluster_data = load_profiles[load_profiles["cluster"] == cluster_label]
        cluster_data.drop(columns=["cluster", "building_name", "user_type"], inplace=True)
        cluster_data.set_index("date", inplace=True)
        if not cluster_data.empty:
            ax[j].plot(cluster_data.T, color='gray', alpha=0.1, linewidth=0.3)
            ax[j].set_title(f"Cluster {cluster_label}")
            ax[j].set_xlabel("Ora del giorno")
            ax[j].set_ylabel("Potenza [W]")
            ax[j].set_xticks(range(0, 24, 4))

total_load_profiles = pd.concat(building_data_list)
centroid_load_profiles = total_load_profiles[["hour", "Net_norm", "cluster"]].groupby(["hour", "cluster"]).median()
centroid_load_profiles.reset_index(inplace=True)
q1 = total_load_profiles[["hour", "Net_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.25)
q1.reset_index(inplace=True)
q3 = total_load_profiles[["hour", "Net_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.75)
q3.reset_index(inplace=True)

palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
for j, cluster_label in enumerate(cluster_labels):
    cluster_data = centroid_load_profiles[centroid_load_profiles["cluster"] == cluster_label]
    cluster_q1 = q1[q1["cluster"] == cluster_label]
    cluster_q3 = q3[q3["cluster"] == cluster_label]

    color = palette[j]
    ax[j].plot(cluster_data.set_index("hour")["Net_norm"], color=color, label="Centroid Consumer")
    ax[j].fill_between(cluster_q1["hour"], cluster_q1.set_index("hour")["Net_norm"], cluster_q3.set_index("hour")["Net_norm"], color=color, alpha=0.3)

plt.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=4)
plt.suptitle(f"Profili di carico dei CONSUMER nei cluster per {aggregate.title()}")
plt.show()

df_sf_consumer = df_sf_total[df_sf_total["user_type"] == "consumer"].drop(columns=["user_type", "building_name"])
df_sf_consumer_melt = df_sf_consumer.melt(id_vars='cluster', var_name='variable', value_name='value')
variables = df_sf_consumer_melt['variable'].unique()

cluster_colors = sns.color_palette("colorblind", n_colors=len(cluster_labels))
cluster_labels = [f'Cluster {i}' for i in range(1, len(cluster_labels) + 1)]
fig, axes = plt.subplots(8, 2, figsize=(20, 30), sharex=True)
axes = axes.flatten()
for i, variable in enumerate(variables):
    ax = axes[i]
    subset = df_sf_consumer_melt[df_sf_consumer_melt['variable'] == variable]
    sns.kdeplot(data=subset, x='value', hue='cluster', fill=True, common_norm=False, alpha=0.5, linewidth=1.5, ax=ax,
                legend=False, palette=cluster_colors)
    ax.set_title(variable)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xlim(-2, 2)
patches = [mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]) for i in range(len(cluster_labels))]
fig.legend(handles=patches, loc='lower center', ncol=len(cluster_labels), title='')
plt.suptitle("Distribuzione dei fattori di forma per i cluster dei consumer", fontsize=18)
plt.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=5)
plt.show()

"""
Prosumer analysis
"""

cluster_labels = np.sort(cluster[cluster["user_type"] != "consumer"]["cluster"].unique())

building_data_list = []
df_cluster_percentage = pd.DataFrame(columns=cluster_labels,
                                     index=[building.building_info["name"] for building in building_list if building.building_info["user_type"] != "consumer"])
df_cluster_count = pd.DataFrame(columns=cluster_labels,
                                index=[building.building_info["name"] for building in building_list if building.building_info["user_type"] != "consumer"])
for building in building_list:
    if building.building_info["user_type"] != "consumer":
        data = building.energy_meter.data.copy()
        data.set_index("timestamp", inplace=True)
        data = data.resample("1H").mean()
        data["building_name"] = building.building_info["name"]
        data["date"] = data.index.date
        data["hour"] = data.index.strftime("%H:%M")
        data.reset_index(inplace=True)
        data["Net_norm"] = data["Net"].clip(lower=0)
        data["Net_norm"] = data["Net_norm"] / data.groupby("date")["Net"].transform("max")

        data = data.merge(cluster, on=["building_name", "date"], how="left")
        data["user_type"] = building.building_info["user_type"]
        building_data_list.append(data)

        building_cluster = cluster[cluster["building_name"] == building.building_info["name"]]
        cluster_counts = building_cluster["cluster"].value_counts().reindex(cluster_labels).fillna(0)
        df_cluster_count.loc[building.building_info["name"]] = cluster_counts
        cluster_percentage = (cluster_counts / cluster_counts.sum() * 100).round(1)
        df_cluster_percentage.loc[building.building_info["name"]] = cluster_percentage

df_cluster_percentage = df_cluster_percentage.astype(float)
df_cluster_count = df_cluster_count.astype(int)

palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
fig, ax = plt.subplots(figsize=(10, 8))
bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
ax.set_xlabel("Percentuale [%]")
ax.set_ylabel("Edificio")
ax.set_title(f"Percentuale di profili di carico dei PROSUMER nei cluster per {aggregate.title()}")
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
plt.show()

palette = sns.color_palette("RdBu", n_colors=len(df_cluster_count.index))
fig, ax = plt.subplots(figsize=(10, 8))
bars = df_cluster_count.T.plot(kind="bar", stacked=True, ax=ax, color=palette)
ax.set_ylabel("Conteggio")
ax.set_xlabel("Cluster")
ax.set_title(f"Conteggio di profili di carico dei PROSUMER nei cluster per {aggregate.title()}")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_cluster_count.index),
            fancybox=True, shadow=True)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()

fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=len(cluster_labels))
unique_labels = []

for i, building in enumerate(building_data_list):

    load_profiles = building[["date", "hour", "Net_norm"]].pivot_table(index="date", columns="hour", values="Net_norm")
    cluster_dates = cluster[cluster["building_name"] == building_list[i].building_info["name"]]
    load_profiles = load_profiles.merge(cluster_dates, on="date", how="left")
    building_name = building_list[i].building_info["name"]

    for j, cluster_label in enumerate(cluster_labels):
        cluster_data = load_profiles[load_profiles["cluster"] == cluster_label]
        cluster_data.drop(columns=["cluster", "building_name", "user_type"], inplace=True)
        cluster_data.set_index("date", inplace=True)
        if not cluster_data.empty:
            ax[j].plot(cluster_data.T, color='gray', alpha=0.1)
            ax[j].set_title(f"Cluster {cluster_label}")
            ax[j].set_xlabel("Ora del giorno")
            ax[j].set_ylabel("Potenza [W]")
            ax[j].set_xticks(range(0, 24, 4))

total_load_profiles = pd.concat(building_data_list)
centroid_load_profiles = total_load_profiles[["hour", "Net_norm", "cluster"]].groupby(["hour", "cluster"]).median()
centroid_load_profiles.reset_index(inplace=True)
q1 = total_load_profiles[["hour", "Net_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.25)
q1.reset_index(inplace=True)
q3 = total_load_profiles[["hour", "Net_norm", "cluster"]].groupby(["hour", "cluster"]).quantile(0.75)
q3.reset_index(inplace=True)

palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
for j, cluster_label in enumerate(cluster_labels):
    cluster_centroid = centroid_load_profiles[centroid_load_profiles["cluster"] == cluster_label]
    cluster_q1 = q1[q1["cluster"] == cluster_label]
    cluster_q3 = q3[q3["cluster"] == cluster_label]

    color = palette[j]
    ax[j].plot(cluster_centroid.set_index("hour")["Net_norm"], color=color, label="Centroid Prosumer")
    ax[j].fill_between(cluster_q1["hour"], cluster_q1.set_index("hour")["Net_norm"], cluster_q3.set_index("hour")["Net_norm"], color=color, alpha=0.3)

plt.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=4)
plt.suptitle(f"Profili di carico dei PROSUMER nei cluster per {aggregate.title()}")
plt.show()

df_sf_prosumer = df_sf_total[df_sf_total["user_type"] != "consumer"].drop(columns=["user_type", "building_name"])
df_sf_prosumer_melt = df_sf_prosumer.melt(id_vars='cluster', var_name='variable', value_name='value')
variables = df_sf_prosumer_melt['variable'].unique()

cluster_colors = sns.color_palette("colorblind", n_colors=len(cluster_labels))
cluster_labels = [f'Cluster {i}' for i in range(1, len(cluster_labels) + 1)]
fig, axes = plt.subplots(8, 2, figsize=(20, 30), sharex=True)
axes = axes.flatten()
for i, variable in enumerate(variables):
    ax = axes[i]
    subset = df_sf_prosumer_melt[df_sf_prosumer_melt['variable'] == variable]
    sns.kdeplot(data=subset, x='value', hue='cluster', fill=True, common_norm=False, alpha=0.5, linewidth=1.5, ax=ax,
                legend=False, palette=cluster_colors)
    ax.set_title(variable)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xlim(-2, 2)
patches = [mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]) for i in range(len(cluster_labels))]
fig.legend(handles=patches, loc='lower center', ncol=5, title='')
plt.suptitle("Distribuzione dei fattori di forma per i cluster dei prosumer", fontsize=18)
plt.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=5)
plt.show()

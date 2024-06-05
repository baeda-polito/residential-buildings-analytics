import os
import pandas as pd
import matplotlib.pyplot as plt
from src.building import load_anguillara, load_garda
from settings import PROJECT_ROOT

building_list = load_garda()

df_list = []
for building in building_list:
    data = building.energy_meter.data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    df_list.append(data)

combined_df = pd.concat(df_list)
data_aggregate = combined_df.groupby('timestamp').sum().reset_index()

fig, ax = plt.subplots(figsize=(20, 5))
ax.fill_between(data_aggregate["timestamp"], data_aggregate["Production"], color="orange", alpha=0.5, label="Production")
ax.plot(data_aggregate["timestamp"], data_aggregate["Load"], color="blue", label="Load")
ax.fill_between(data_aggregate["timestamp"], data_aggregate["Net"], color="green", alpha=0.5, label="Net")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Power (W)", fontsize=12)
ax.set_title("Anguillara aggregate", fontsize=18)
fig.legend(bbox_to_anchor=(1, 0.5), loc='center right', fontsize=12, ncol=1, fancybox=True, shadow=True)
plt.savefig(os.path.join(PROJECT_ROOT, "figures", "aggregate_garda.png"))
plt.show()

time_range = pd.date_range(start="2024-03-01T00:00:00Z", end=data["timestamp"].max().strftime('%Y-%m-%dT%H:%M:%SZ'),
                           freq="15min")

plt.figure(figsize=(15, 5))
cumulative_sum = pd.Series([0] * len(time_range), index=time_range)
for i, df in enumerate(df_list):
    data = df.merge(pd.DataFrame(time_range, columns=["timestamp"]), on="timestamp", how="right")
    data.drop_duplicates(subset=["timestamp"], inplace=True)
    net_column = data['Net']
    net_column.index = data['timestamp']
    plt.fill_between(data['timestamp'], cumulative_sum, cumulative_sum + net_column, label=f"DU_{i+1}")
    cumulative_sum += net_column

plt.xlim(time_range[0], time_range[-1])
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Power [W]', fontsize=14)
plt.title('Stacked Area Plot for Anguillara aggregate', fontsize=18)
plt.legend(bbox_to_anchor=(0.5, -0.28), loc='lower center', fontsize=12, ncol=10, fancybox=True, shadow=True)
plt.tight_layout(rect=(0, 0.01, 1, 1))
plt.savefig(os.path.join(PROJECT_ROOT, "figures", "aggregate", "stacked_area_plot_garda.png"))
plt.show()
plt.close(fig)

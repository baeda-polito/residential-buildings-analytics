import numpy as np
import matplotlib.pyplot as plt

from building import load_anguillara


if __name__ == "__main__":
    build_dict = load_anguillara()

    # Find the common timestamps
    timestamps = [building.energy_meter.data.timestamp.values for building in build_dict]
    old_time = timestamps[0]

    for stamp in timestamps:
        new_time = np.intersect1d(old_time, stamp)
        old_time = new_time

    times = old_time
    n_steps = len(times)

    # Initialize the variables
    total_production = np.zeros(n_steps)
    total_consumption = np.zeros(n_steps)
    total_withdrawn = np.zeros(n_steps)
    total_injection = np.zeros(n_steps)

    for build in build_dict:

        print(build.building_info['name'])
        # Find the indeces of the common timestamps
        indeces = np.isin(build.energy_meter.data['timestamp'].values, times)

        # Aggregate the data for consumption and withdrawal
        total_consumption += build.energy_meter.data.loc[indeces]["Load"].values
        total_withdrawn += build.energy_meter.data.loc[indeces]["Net"].clip(lower=0).values

        if build.building_info['user_type'] != "consumer":

            # Aggregate the data for production and injection
            total_production += build.energy_meter.data.loc[indeces]['Production'].values
            total_injection += build.energy_meter.data.loc[indeces]["Net"].clip(upper=0).values

    # Compute the KPIs
    se_series = np.min((-total_injection, total_withdrawn), axis=0)
    se_tot = np.sum(se_series)

    # How much of the injected energy is shared within the community
    virtual_sc = se_tot / np.sum(-total_injection)

    # How much of the withdrawn energy comes from prosumer surplus
    virtual_ss = se_tot / np.sum(total_withdrawn)

    # How much of the consumed energy is self-produced
    self_consumed_energy = np.sum(np.min((total_production, total_consumption), axis=0))

    ss = self_consumed_energy / np.sum(total_consumption)

    # How much of the produced energy is self-consumed
    sc = self_consumed_energy / np.sum(total_production)

    # Other KPIs from papers
    print(f"Total Shared Energy: {se_tot/1000}")
    print(f"Total Withdrawn: {np.sum(total_withdrawn/1000)}")
    print(f"Total Injection: {np.sum(-total_injection/1000)}")
    print(f"Virtual Self-Consumption: {virtual_sc}")
    print(f"Virtual Self-Sufficiency: {virtual_ss}")
    print(f"Self-Consumed energy: {self_consumed_energy/1000}")
    print(f"Total Consumption: {np.sum(total_consumption/1000)}")
    print(f"Total Production: {np.sum(total_production/1000)}")
    print(f"Self-Consumption: {sc}")
    print(f"Self-Sufficiency: {ss}")

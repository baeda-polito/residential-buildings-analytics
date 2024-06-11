import numpy as np
# import matplotlib.pyplot as plt

from building import load_anguillara


if __name__ == "__main__":
    build_dict = load_anguillara()

    # Find the common timestamps
    timestamps = [building.energy_meter.data.timestamp.values for building in build_dict]
    old_time = timestamps[0]

    for stamp in timestamps:
        new_time = np.intersect1d(old_time, stamp)
        old_time = new_time

    times = new_time
    n_steps = len(times)
    n_days = (new_time[-1]-new_time[0]) / np.timedelta64(1, 'D')
    #
    # Get indeces of midnigths to compute daily peak and PAR
    time_idx = []
    for idx, get_time in enumerate(new_time):
        time_of_day = str(get_time).split('T')[1]
        if time_of_day[:8] == '00:00:00':
            time_idx.append(idx)

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

    daily_peaks = []
    daily_peaks_w = []
    pars = []
    pars_w = []

    for start, end in zip(time_idx, time_idx[1:]):

        peaks = np.max(total_consumption[start:end])
        peaks_w = np.max(total_withdrawn[start:end])

        means = np.mean(total_consumption[start:end])
        means_w = np.mean(total_withdrawn[start:end])

        daily_peaks.append(peaks)
        daily_peaks_w.append(peaks_w)

        pars.append(peaks / means)
        pars_w.append(peaks_w / means_w)

    # Compute the average daily peak and PAR
    daily_peak = np.mean(daily_peaks)
    daily_peak_w = np.mean(daily_peaks_w)

    average_par = np.mean(pars)
    average_par_w = np.mean(pars_w)

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
    # print(f"Total Shared Energy: {se_tot/1000}")
    # print(f"Total Withdrawn: {np.sum(total_withdrawn/1000)}")
    # print(f"Total Injection: {np.sum(-total_injection/1000)}")
    # print(f"Virtual Self-Consumption: {virtual_sc}")
    # print(f"Virtual Self-Sufficiency: {virtual_ss}")
    # print(f"Self-Consumed energy: {self_consumed_energy/1000}")
    # print(f"Total Consumption: {np.sum(total_consumption/1000)}")
    # print(f"Total Production: {np.sum(total_production/1000)}")

    print(f"Total Shared Energy per day: {se_tot/1000 / n_days}")
    print(f"Total Withdrawn per day: {np.sum(total_withdrawn/1000) / n_days}")
    print(f"Total Injection per day: {np.sum(-total_injection/1000) / n_days}")
    print(f"Virtual Self-Consumption: {virtual_sc}")
    print(f"Virtual Self-Sufficiency: {virtual_ss}")
    print(f"Self-Consumed per day energy: {self_consumed_energy/1000 / n_days}")
    print(f"Total Consumption per day: {np.sum(total_consumption/1000) / n_days}")
    print(f"Total Production per day: {np.sum(total_production/1000) / n_days}")
    print(f"Self-Consumption: {sc}")
    print(f"Self-Sufficiency: {ss}")
    print(f"Average daily peak considering REC electrical load: {daily_peak}")
    print(f"Average daily peak considering REC electrical withdrawn: {daily_peak_w}")
    print(f"Average daily Peak-to-Average ratio considering REC electrical load: {average_par}")
    print(f"Average daily Peak-to-Average ratio considering REC electrical withdrawn: {average_par_w}")

import gc
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import sys
import os
from tqdm import tqdm

# Add the root directory to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from Implementation.Utils.__utils__ import timer



def load_data(data_name):
    """Loads and formats data from manually specified file paths"""
    base_path = "../Prepared_Data/"
    file_paths = {
        "train": "./train.feather",
        "test": "./test.feather",
        "meta": "./building_metadata.feather",
        "weather_train": "./weather_train.feather",
        "weather_test": "./weather_test.feather"
    }
    return pd.read_feather(base_path+file_paths[data_name])

def reduceDataSet(train, test, building_meta, weather_train, weather_test):
    """Keep only the first 290 buildings by building id. It also prints how many
    rows were removed every time it is called."""
    numRows = len(train)
    train = train[train.building_id < 290]
    print(f"INFO: Removed {numRows - len(train)} rows from train dataset.")
    numRows = len(test)
    test = test[test.building_id < 290]
    print(f"INFO: Removed {numRows - len(test)} rows from test dataset.")

    numRows = len(building_meta)
    building_meta = building_meta[building_meta.building_id < 290]
    print(f"INFO: Removed {numRows - len(building_meta)} rows from building metadata.")

    # The first 290 buildings actually exists in site_id 1 and site_id 2.
    # So remove all site_id > 2 from weather_train and weather_test
    numRows = len(weather_train)
    weather_train = weather_train[weather_train.site_id < 3]
    print(f"INFO: Removed {numRows - len(weather_train)} rows from train weather dataset.")
    numRows = len(weather_test)
    weather_test = weather_test[weather_test.site_id < 3]
    print(f"INFO: Removed {numRows - len(weather_test)} rows from test weather dataset.")

    return train, test, building_meta, weather_train, weather_test

def reduceDataSetTo_5_Buildings(train, test, building_meta, weather_train, weather_test):
    """Keep only the first 5 buildings for 2 sites. 
    Prints how many rows were removed and the final shape of each DataFrame."""

    # Pick up 2 sites
    selected_sites = [0, 1]

    # Get 5 building ids for each site
    selected_buildings = building_meta[building_meta.site_id.isin(selected_sites)]
    selected_buildings = (
        selected_buildings.groupby("site_id")
        .head(5)  # Pick first 5 buildings per site
        .building_id
        .tolist()
    )

    # Filter train
    original_rows = len(train)
    train = train[train.building_id.isin(selected_buildings)]
    removed = original_rows - len(train)
    print(f"INFO: Removed {removed} rows from train dataset ({removed} / {original_rows})")

    # Filter test
    original_rows = len(test)
    test = test[test.building_id.isin(selected_buildings)]
    removed = original_rows - len(test)
    print(f"INFO: Removed {removed} rows from test dataset ({removed} / {original_rows})")

    # Filter building metadata
    original_rows = len(building_meta)
    building_meta = building_meta[building_meta.building_id.isin(selected_buildings)]
    removed = original_rows - len(building_meta)
    print(f"INFO: Removed {removed} rows from building metadata ({removed} / {original_rows})")

    # Filter train weather
    original_rows = len(weather_train)
    weather_train = weather_train[weather_train.site_id.isin(selected_sites)]
    removed = original_rows - len(weather_train)
    print(f"INFO: Removed {removed} rows from train weather dataset ({removed} / {original_rows})")

    # Filter test weather
    original_rows = len(weather_test)
    weather_test = weather_test[weather_test.site_id.isin(selected_sites)]
    removed = original_rows - len(weather_test)
    print(f"INFO: Removed {removed} rows from test weather dataset ({removed} / {original_rows})")

    # Final shapes
    print("\nFinal DataFrame Shapes:")
    print(f"  Train Dataset: {train.shape}")
    print(f"  Test Dataset: {test.shape}")
    print(f"  Building Metadata: {building_meta.shape}")
    print(f"  Train Weather: {weather_train.shape}")
    print(f"  Test Weather: {weather_test.shape}")

    return train, test, building_meta, weather_train, weather_test

# Define groupings and corresponding priors
groups_and_priors = {
    ("hour",):        None,
    ("weekday",):     None,
    ("month",):       None,
    ("building_id",): None,
    ("primary_use",): None,
    ("site_id",):     None,    
    ("meter",):       None,
    ("meter", "hour"):        ["gte_meter", "gte_hour"],
    ("meter", "weekday"):     ["gte_meter", "gte_weekday"],
    ("meter", "month"):       ["gte_meter", "gte_month"],
    ("meter", "building_id"): ["gte_meter", "gte_building_id"],
    ("meter", "primary_use"): ["gte_meter", "gte_primary_use"],
    ("meter", "site_id"):     ["gte_meter", "gte_site_id"],
    ("meter", "building_id", "hour"):    ["gte_meter_building_id", "gte_meter_hour"],
    ("meter", "building_id", "weekday"): ["gte_meter_building_id", "gte_meter_weekday"],
    ("meter", "building_id", "month"):   ["gte_meter_building_id", "gte_meter_month"],
}

def process_timestamp(df): 
    df.timestamp = pd.to_datetime(df.timestamp)
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600


def process_weather(df, dataset, fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    if fix_timestamps:
        site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)

    if interpolate_na:
        site_dfs = []
        unique_sites = df.site_id.unique()
        
        for site_id in tqdm(unique_sites, desc="Processing Weather Data", unit="site"):
            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(
                range(8784 if dataset == "train" else 8784, 26304)
            )
            site_df.site_id = site_id
            for col in tqdm([c for c in site_df.columns if c != "site_id"], desc=f"Interpolating Site {site_id}", leave=False):
                if add_na_indicators:
                    site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(
                    limit_direction="both", method="spline", order=3
                ).fillna(df[col].median())

            site_dfs.append(site_df)

        df = pd.concat(site_dfs).reset_index()

    if add_na_indicators:
        for col in tqdm(df.columns, desc="Adding NA Indicators"):
            if df[col].isna().any():
                df[f"had_{col}"] = ~df[col].isna()

    return df.fillna(-1)

def add_lag_feature(df, window=3, group_cols="site_id", lag_cols=["air_temperature"]):
    rolled = df.groupby(group_cols)[lag_cols].rolling(window=window, min_periods=0, center=True)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.quantile(0.95).reset_index().astype(np.float16)
    lag_min = rolled.quantile(0.05).reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in lag_cols:
        df[f"{col}_mean_lag{window}"] = lag_mean[col]
        df[f"{col}_max_lag{window}"] = lag_max[col]
        df[f"{col}_min_lag{window}"] = lag_min[col]
        df[f"{col}_std_lag{window}"] = lag_std[col]

def add_features(df):
    df["hour"] = df.ts.dt.hour
    df["weekday"] = df.ts.dt.weekday
    df["month"] = df.ts.dt.month
    df["year"] = df.ts.dt.year    
    df["weekday_hour"] = df.weekday.astype(str) + "-" + df.hour.astype(str)
    df["hour_x"] = np.cos(2*np.pi*df.timestamp/24)
    df["hour_y"] = np.sin(2*np.pi*df.timestamp/24)
    df["month_x"] = np.cos(2*np.pi*df.timestamp/(30.4*24))
    df["month_y"] = np.sin(2*np.pi*df.timestamp/(30.4*24))
    df["weekday_x"] = np.cos(2*np.pi*df.timestamp/(7*24))
    df["weekday_y"] = np.sin(2*np.pi*df.timestamp/(7*24))
    df["year_built"] = df["year_built"]-1900
    bm_ = df.building_id.astype(str) + "-" + df.meter.astype(str) + "-"
    df["building_weekday_hour"] = bm_ + df.weekday_hour
    df["building_weekday"] = bm_ + df.weekday.astype(str)
    df["building_month"] = bm_ + df.month.astype(str)
    df["building_hour"] = bm_ + df.hour.astype(str)    
    df["building_meter"] = bm_
    dates_range = pd.date_range(start="2015-12-31", end="2019-01-01")
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())    
    df["is_holiday"] = df.ts.dt.normalize().astype("datetime64[ns]").isin(us_holidays).astype(np.int8)

def printInfo(train, test, weather_train, weather_test, building_meta):
    # Print the dataset information
    print("Train dataset:")
    print(train.info())
    print("Test dataset:")
    print(test.info())
    print("Building Metadata:")
    print(building_meta.info())
    print("Train Weather:")
    print(weather_train.info())
    print("Test Weather:")
    print(weather_test.info())

def print_usage():
    print("Usage:")
    print("  python your_script.py           # Run full dataset")
    print("  python your_script.py --faster  # Reduce to 290 buildings")
    print("  python your_script.py --tiny    # Reduce to 10 buildings (5 from 2 sites)")


if __name__ == "__main__":

    print_usage()

    # Loading the data
    with timer("Loading data"):
        train = load_data("train")
        test = load_data("test")
        building_meta = load_data("meta")
        train_weather = load_data("weather_train")
        test_weather = load_data("weather_test")

    # printInfo(train, test, train_weather, test_weather, building_meta)


    # Reduce the dataset for faster pre-processing. But only if faster flag is true
    with timer("Dataset Reduction for testing pre-processing"):
        reduce= False
        n = len(sys.argv)
        if(n>1):
            if(sys.argv[1]=="--faster"):
                reduce = True
                print("Reducing Dataset option enabled. Reducing dataset for faster pre-processing.")
                train, test, building_meta, train_weather, test_weather = reduceDataSet(train, test, building_meta, train_weather, test_weather)
                print("Dataset reduced.")
            elif(sys.argv[1]=="--tiny"):
                print("Reducing Dataset: Keeping 5 buildings from 2 sites.")
                train, test, building_meta, train_weather, test_weather = reduceDataSetTo_5_Buildings(
                    train, test, building_meta, train_weather, test_weather
                )
                print("Tiny dataset reduction complete.")
            else:
                print("Invalid argument")
                print_usage()
                sys.exit()
        else:
            print("Reducing Dataset option not enabled. Proceeding with full dataset.")

        # if reduce:
        #     printInfo(train, test, train_weather, test_weather, building_meta)
    
    # Pre-processing
    print(f"INFO: Pre-processing started.")
    print(f"UPDATE: Processing timestamp.")
    train["ts"] = pd.to_datetime(train.timestamp)
    test["ts"] = pd.to_datetime(test.timestamp)
    with timer("Processing TimeStamps"):
        process_timestamp(train)
        process_timestamp(test)
        process_timestamp(train_weather)
        process_timestamp(test_weather)

    print(f"UPDATE: Processing weather.")
    with timer("Processing Weather Data"):
        process_weather(train_weather, "train")
        process_weather(test_weather, "test")

    with timer("Adding Lag features"):
        print(f"UPDATE: Adding lag features.")
        for window_size in [7, 73]:
            add_lag_feature(train_weather, window=window_size)
            add_lag_feature(test_weather, window=window_size)

    
    # Merge datasets
    with timer("Merging Datasets"):
        print(f"UPDATE: Merging datasets.")
        train = pd.merge(train, building_meta, "left", "building_id")
        train = pd.merge(train, train_weather, "left", ["site_id", "timestamp"])
        test = pd.merge(test, building_meta, "left", "building_id")
        test = pd.merge(test, test_weather, "left", ["site_id", "timestamp"])

    # Add features
    with timer("Adding Features"):
        print(f"UPDATE: Adding features.")
        add_features(train)
        add_features(test)

    with timer("Free up memory"):
        del train_weather, test_weather
        gc.collect()
    
    with timer("Remove unnecessary columns"):
        train.drop(["ts"], axis=1, inplace=True)
        test.drop(["ts"], axis=1, inplace=True)


    gc.collect()
    train.info()
    test.info()

    # Save processed data
    with timer("Saving Processed Data"):
        train.to_feather("../Processed_Data/train_processed.feather")
        test.to_feather("../Processed_Data/test_processed.feather")
    
    print("Preprocessing complete! Processed data saved.")
    gc.collect()

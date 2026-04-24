import pandas as pd

from ass1.config import DATA_DIR
from ass1.modeling.location import WindResourceData
from ass1.modeling.turbine import WindTurbine

TURBINE_DATA = DATA_DIR / "turbine_data"


def load_tmy():
    """Load the Typical Meteorological Year of the site"""
    tmy_path = DATA_DIR / "nasa_tmy.csv"
    with open(tmy_path) as fp:
        skip_until_line = next(i for i, line in enumerate(fp) if line.startswith("YEAR"))

    df = pd.read_csv(tmy_path, skiprows=skip_until_line, dtype_backend="numpy_nullable")
    df["Time"] = pd.to_datetime(
        df[["YEAR", "MO", "DY", "HR"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
        )
    )
    return df


def load_2023() -> pd.DataFrame:
    """Load the Typical Meteorological Year of the site"""
    file_path = DATA_DIR / "nasa_2023.csv"
    with open(file_path) as fp:
        skip_until_line = next(i for i, line in enumerate(fp) if line.startswith("YEAR"))

    df = pd.read_csv(file_path, skiprows=skip_until_line, dtype_backend="numpy_nullable")
    df["Time"] = pd.to_datetime(
        df[["YEAR", "MO", "DY", "HR"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
        )
    )
    return df


def load_demand():
    """Load demand at the site"""
    demand_path = DATA_DIR / "load_project_2026-1.xlsx"
    return pd.read_excel(demand_path, dtype_backend="numpy_nullable")


def load_site_year() -> pd.DataFrame:
    """Load demand and wind speed at the site"""
    demand = load_demand()
    weather = load_2023()

    # Replace TMY year with the demand year so the join keys align
    site_year = demand["Time"].dt.year.iloc[0]
    weather["Time"] = weather["Time"].apply(lambda t: t.replace(year=site_year))

    merged = demand.merge(weather, on="Time", how="inner")
    return merged[["Time", "Load [MW]", "WS50M", "WD50M"]]


def load_wind_resource() -> WindResourceData:
    df = load_2023()
    return WindResourceData(df["Time"], df["WS50M"], df["WD50M"])


def load_nrel_6MW() -> WindTurbine:
    file_name = "2016CACost_NREL_Reference_6MW_155"

    return WindTurbine.from_yaml(
        TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv"
    )


def load_nrel_8MW() -> WindTurbine:
    file_name = "2016CACost_NREL_Reference_8MW_180"

    return WindTurbine.from_yaml(
        TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv"
    )


def load_nrel_10MW() -> WindTurbine:
    file_name = "2016CACost_NREL_Reference_10MW_205"

    return WindTurbine.from_yaml(
        TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv"
    )


def load_dtu_10MW() -> WindTurbine:
    file_name = "DTU_Reference_v1_10MW_178"

    return WindTurbine.from_yaml(
        TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv"
    )

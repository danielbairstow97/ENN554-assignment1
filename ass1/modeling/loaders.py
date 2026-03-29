import polars as pl

from ass1.config import DATA_DIR
from ass1.modeling.turbine import Turbine

TURBINE_DATA = DATA_DIR / "turbine_data"


def load_tmy():
    tmy_path = DATA_DIR / "nasa_tmy.csv"
    with open(tmy_path) as fp:
        skip_until_line = next(filter(lambda x: x[1].startswith("YEAR"), enumerate(fp)))[0]

    df = pl.read_csv(tmy_path, skip_rows=skip_until_line)
    df = df.with_columns(
        pl.datetime(
            year=pl.col("YEAR"), month=pl.col("MO"), day=pl.col("DY"), hour=pl.col("HR")
        ).alias("datetime")
    )

    return df


def load_nrel_6MW() -> Turbine:
    file_name = "2016CACost_NREL_Reference_6MW_155"

    return Turbine.from_yaml(TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv")


def load_nrel_8MW() -> Turbine:
    file_name = "2016CACost_NREL_Reference_8MW_180"

    return Turbine.from_yaml(TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv")


def load_nrel_10MW() -> Turbine:
    file_name = "2016CACost_NREL_Reference_10MW_205"

    return Turbine.from_yaml(TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv")


def load_dtu_10MW() -> Turbine:
    file_name = "DTU_Reference_v1_10MW_178"

    return Turbine.from_yaml(TURBINE_DATA / f"{file_name}.yaml", TURBINE_DATA / f"{file_name}.csv")

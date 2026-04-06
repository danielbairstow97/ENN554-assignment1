import pandas as pd
import typer

from ass1.config import OUTPUT_DIR
from ass1.modeling.financial import FinancialModel
from ass1.modeling.loaders import (
    load_dtu_10MW,
    load_nrel_6MW,
    load_nrel_8MW,
    load_nrel_10MW,
    load_wind_resource,
)
from ass1.modeling.location import WindResourceData, WindResourceModel
from ass1.modeling.turbine import WindTurbine

app = typer.Typer()


def assess_wind_resource(resource: WindResourceData):
    # Plot wind rose
    wind_rose_fig = resource.plot_wind_rose()
    wind_rose_fig.write_image(OUTPUT_DIR / "wind_resource/wind_rose.png")

    # Plot wind rose heatmap

    # Plot wind speed distribution


def assess_turbine(turbine: WindTurbine, resource: WindResourceModel):
    turbine_dir = OUTPUT_DIR / f"turbines/{turbine.name}"

    # Plot power curve
    turbine.plot_power_curve().to_image(turbine_dir / "power_curve.png")

    # Plot turbine power distribution
    resource.plot_power_distribution(turbine).to_image(turbine_dir / "power_distribution.png")


def select_turbine(
    turbine_options: list[WindTurbine], resource: WindResourceModel, finance: FinancialModel
) -> WindTurbine:
    results = {}
    for turbine in turbine_options:
        aep = resource.aep(turbine)
        c_rotor, c_tower, c_other = finance.cost_turbine_components(turbine)
        lcoe = finance.lcoe(turbine, aep)

        results[turbine.name] = {
            "Anunual Energy Production (GWh)": aep * 1e-6,
            "Rotor + Tower cost (AUD)": c_rotor + c_tower,
            "Other costs (AUD)": c_other,
            "TCC (AUD)": c_rotor + c_tower + c_other,
            "Levelised Cost of Energy (AUD/MWh)": lcoe * 1e3,
        }

    summary_df = pd.DataFrame(results).T
    summary_df.index.name = "Turbine"
    summary_df.to_csv(OUTPUT_DIR / "wind_turbines/summary.csv")

    # Select turbine
    selected_turbine = summary_df["Levelised Cost of Energy (AUD/MWh)"].idxmax()
    for turbine in turbine_options:
        if turbine.name == selected_turbine:
            return turbine

    raise ValueError(f"{selected_turbine} not a turbine option")


@app.command()
def main(assess=True):
    wind = load_wind_resource()

    turbine_options = [load_nrel_6MW(), load_nrel_8MW(), load_nrel_10MW(), load_dtu_10MW()]

    financial_model = FinancialModel(fcr=0.05)

    if assess:
        assess_wind_resource(wind)
        [assess_turbine(t, wind.create_weibull_wind_resource()) for t in turbine_options]

    


if __name__ == "__main__":
    app()

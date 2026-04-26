import pandas as pd
import plotly.express as px
import typer

from ass1.config import OUTPUT_DIR
from ass1.loaders import (
    load_dtu_10MW,
    load_nrel_6MW,
    load_nrel_8MW,
    load_nrel_10MW,
    load_site_year,
    load_wind_resource,
)
from ass1.modeling.farm import ConfigurationOption as CO
from ass1.modeling.farm import WindFarm
from ass1.modeling.financial import FinancialModel
from ass1.modeling.location import WindResourceData, WindResourceModel
from ass1.modeling.microgrid import Microgrid
from ass1.modeling.turbine import WindTurbine
from ass1.plots import (
    plot_correlation_heatmap,
    plot_load_heatmap,
    plot_load_timeseries,
    plot_seasonal_wind_roses,
)

app = typer.Typer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, path):
    """Save a Plotly figure, skipping if the file already exists."""
    #if path.exists():
    #    typer.echo(f"  [skip] {path.name} already exists")
    #    return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=4)
    typer.echo(f"  [save] {path.name}")


def assess_extra():
    out = OUTPUT_DIR / "load"
    data = load_site_year()
    _save(plot_load_timeseries(data), out / "timeseries.jpeg")
    _save(plot_load_heatmap(data), out / "heatmap.jpeg")
    _save(plot_correlation_heatmap(data), OUTPUT_DIR / "correlations.jpeg")
    _save(plot_seasonal_wind_roses(data), OUTPUT_DIR / "wind_resource/seasonal_windrose.jpeg")


# ---------------------------------------------------------------------------
# Assessment steps
# ---------------------------------------------------------------------------
def assess_wind_resource(wind: WindResourceData) -> None:
    """Characterise the site wind resource and write plots to disk."""
    typer.echo("── Wind resource assessment")
    out = OUTPUT_DIR / "wind_resource"

    weibull_params = wind.create_weibull()

    _save(wind.plot_wind_speed(rolling_window=7 * 24), out / "wind_speed_ts.jpeg")
    _save(wind.plot_wind_distribution(weibull_params), out / "distribution.jpeg")
    _save(wind.plot_wind_rose(), out / "wind_rose.jpeg")
    _save(wind.plot_wind_rose(mode="heatmap"), out / "wind_rose_heatmap.jpeg")
    _save(wind.plot_wind_speed_heatmap(agg="mean"), out / "ws_heatmap_mean.jpeg")


def assess_turbine(turbine: WindTurbine, wind: WindResourceModel) -> None:
    """Plot the power curve and power distribution for a single turbine."""
    typer.echo(f"── Turbine assessment: {turbine.name_str}")
    out = OUTPUT_DIR / "turbines" / turbine.name_str

    _save(turbine.plot_power_curve(), out / "power_curve.jpeg")
    _save(wind.plot_power_distribution(turbine), out / "power_distribution.jpeg")


def compare_turbines(
    fin: FinancialModel,
    wind: WindResourceModel,
    turbine_options: list[WindTurbine],
) -> WindTurbine:
    """
    Compute AEP, CAPEX and LCOE for every candidate turbine.
    Saves a cost breakdown bar chart and a summary CSV, then returns
    the turbine with the lowest LCOE.
    """
    typer.echo("── Turbine comparison")
    out = OUTPUT_DIR / "turbines"

    capex_rows = {}
    summary_rows = {}

    for turbine in turbine_options:
        aep = wind.aep(turbine)  # kWh/yr
        capex, cost_breakdown = fin.cost_turbine(turbine)  # AUD
        lcoe = fin.lcoe(turbine, aep)  # AUD/kWh

        capex_rows[turbine.name_str] = cost_breakdown
        summary_rows[turbine.name_str] = {
            "Annual Energy Production (GWh)": aep * 1e-6,
            "Total Capital Cost (AUD)": capex,
            "Levelised Cost of Energy (AUD/MWh)": lcoe * 1e3,
        }

    # CAPEX cost breakdown bar chart
    capex_df = pd.DataFrame(capex_rows).T.rename_axis("Turbine")
    _save(
        px.bar(
            capex_df.reset_index(),
            x="Turbine",
            y=capex_df.columns,
            title="Turbine CAPEX Breakdown",
            labels={"value": "CAPEX (AUD)", "variable": "Component"},
        ),
        out / "capex_breakdown.jpeg",
    )

    # Summary table
    summary_df = pd.DataFrame(summary_rows).T.rename_axis("Turbine")
    csv_path = out / "comparison.csv"
    if not csv_path.exists():
        out.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(csv_path)
        typer.echo("  [save] comparison.csv")

    # Select turbine with lowest LCOE
    best_name = summary_df["Levelised Cost of Energy (AUD/MWh)"].idxmin()
    typer.echo(f"  Best turbine: {best_name}")

    for turbine in turbine_options:
        if turbine.name_str == best_name:
            return turbine

    raise ValueError(f"Selected turbine '{best_name}' not found in options.")


def assess_windfarm(wind: WindResourceModel, farm: WindFarm) -> None:
    """Placeholder — add farm-level plots and metrics here."""
    typer.echo(f"── Wind farm assessment: {farm.configuration}")
    out = OUTPUT_DIR / "farm" / str(farm.configuration)
    out.mkdir(parents=True, exist_ok=True)
    typer.echo(f"  Optimising layout...")
    farm.optimize_layout(wind)

    typer.echo(f"  Prevailing wind direction: {farm._prevailing_wind_direction():.1f}°")

    _save(farm.plot_aep_per_turbine(wind), out / "aep_per_turbine.jpeg")

def compare_microgrids(fin: FinancialModel, turbine: WindTurbine) -> None:
    """Solve the microgrid network for each farm configuration."""
    typer.echo("── Microgrid comparison")
    site_df = load_site_year()

    results = {}
    for configuration in [CO.NINE, CO.SIXTEEN, CO.TWENTY_FIVE]:
        typer.echo(f"  configuration: {configuration}")
        farm = WindFarm(configuration, turbine)
        mg = Microgrid(farm)
        mg.prepare_network(site_df)
        battery_size = mg.solve_network()

        results[configuration] = {
            "Nameplate Capacity (MW)": farm.nameplate_capacity_mw,
            "Battery Size": battery_size,
            "Farm Cost (AUD)": fin.cost_windfarm(farm),
        }

    results_df = pd.DataFrame(results).T.rename_axis("Configuration")
    typer.echo(results_df.to_string())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@app.command()
def main(
    assess: bool = typer.Option(
        True, "--assess/--no-assess", help="Run wind + turbine assessments"
    ),
    compare: bool = typer.Option(True, "--compare/--no-compare", help="Run turbine comparison"),
    farm: bool = typer.Option(True, "--farm/--no-farm", help="Run microgrid comparison"),
) -> None:
    fin = FinancialModel(fcr=0.05)
    wind_data = load_wind_resource()
    wind_resource = wind_data.create_wind_model()
    turbine_options = [load_nrel_6MW(), load_nrel_8MW(), load_nrel_10MW(), load_dtu_10MW()]

    if assess:
        assess_extra()
        assess_wind_resource(wind_data)
        for turbine in turbine_options:
            assess_turbine(turbine, wind_resource)

    if compare:
        best_turbine = compare_turbines(fin, wind_resource, turbine_options)
    else:
        # Fall back to lowest-LCOE without re-running if compare is skipped
        # — replace with a cached result or prompt if needed
        raise typer.Exit("Re-run with --compare to select a turbine before --farm.")

    if farm:
        for config in CO:
            wf = WindFarm(config, best_turbine, wind_data)
            assess_windfarm(wind_resource, wf)
        compare_microgrids(fin, best_turbine)


if __name__ == "__main__":
    app()

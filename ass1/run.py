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
from ass1.modeling.farm import WindFarm, WindFarmOptimiser
from ass1.modeling.financial import MicrogridFinancialModel
from ass1.modeling.location import WindResourceData, WindResourceModel
from ass1.modeling.microgrid import Microgrid
from ass1.modeling.turbine import WindTurbine
from ass1.plots import (
    plot_correlation_heatmap,
    plot_load_heatmap,
    plot_load_timeseries,
    plot_power_analysis,
    plot_power_curves,
    plot_seasonal_wind_roses,
)

app = typer.Typer()

FORCE_SAVE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, path):
    """Save a Plotly figure, skipping if the file already exists."""
    if path.exists() and not FORCE_SAVE:
        typer.echo(f"  [skip] {path.name} already exists")
        return
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
    fin: MicrogridFinancialModel,
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
    details = {}

    for turbine in turbine_options:
        aep = wind.aep(turbine)  # kWh/yr
        capex, cost_breakdown = fin.cost_turbine_capex(turbine)  # AUD
        lcoe = fin.lcoe(turbine, aep)  # AUD/kWh

        details[turbine.name_str] = {
            "Rated Wind Speed (m/s)": turbine.rated_wind_speed,
            "Rated Power (kW)": turbine.rated_power_kw,
            "Hub Height (m)": turbine.extra_hub_height,
            "Rotor Area (m2)": turbine.rotor_area,
            "Cut in Wind Speed (m/s)": turbine.cut_in_wind_speed,
            "Cut out Wind Speed (m/s)": turbine.cut_out_wind_speed,
        }

        capex_rows[turbine.name_str] = cost_breakdown
        summary_rows[turbine.name_str] = {
            "Annual Energy Production (GWh)": aep * 1e-6,
            "Rotor + Tower (AUD)": cost_breakdown["Rotor + Tower (AUD)"],
            "Other (AUD)": cost_breakdown["Other (AUD)"],
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
    # Details table
    details_path = out / "details.csv"
    details_df = pd.DataFrame(details).rename_axis("Parameter")
    if not details_path.exists():
        details_df.to_csv(details_path)
        typer.echo("  [save] details.csv")

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

    # Plot comparisons
    _save(plot_power_curves(turbine_options), out / "power_curves.jpg")
    _save(plot_power_analysis(wind, turbine_options), out / "power_analysis.jpg")

    for turbine in turbine_options:
        if turbine.name_str == best_name:
            return turbine

    raise ValueError(f"Selected turbine '{best_name}' not found in options.")


def compare_microgrids(
    fin: MicrogridFinancialModel, options: dict[CO, WindFarm], wind: WindResourceModel
) -> None:
    """Solve the microgrid network for each farm configuration."""
    typer.echo("── Microgrid comparison")
    site_df = load_site_year()

    out = OUTPUT_DIR / "farm"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for configuration, farm in options.items():
        farm_dir = out / str(configuration)

        typer.echo(f"  configuration: {configuration}")
        mg = Microgrid(farm)
        typer.echo("  preparing network")
        mg.prepare_network(site_df, wind)
        typer.echo("  solving network")
        battery_size_mwh, battery_power_mwh = mg.solve_network()
        typer.echo(f"  network solved: Slack Generated {mg.slack_energy_mwh}")

        _save(mg.plot_battery_soc(), farm_dir / "mg_battery.jpg")
        _save(mg.plot_dispatch(), farm_dir / "mg_dispatch.jpg")

        _, capex = fin.cost_microgrid_capex(
            farm,
            battery_power_mwh * 1e3,
            battery_size_mwh * 1e3,
        )

        results[configuration] = {
            "Nameplate Capacity (MW)": farm.nameplate_capacity_mw,
            "BESS Capacity (MWh)": battery_size_mwh,
            "BESS Power (MW)": battery_power_mwh,
            "BESS CAPEX (AUD)": capex["BESS CAPEX (AUD)"],
            "Wind Farm CAPEX (AUD))": capex["Wind Farm CAPEX (AUD)"],
            "Annual Cost of Energy (AUD/kWh)": fin.acoe(
                farm,
                battery_power_mwh * 1e3,
                battery_size_mwh * 1e3,
                site_df["Load [MW]"].sum() * 1e3,
            ),
        }

    results_df = pd.DataFrame(results).T.rename_axis("Configuration")
    results_df.to_csv(out / "mg_comparison.csv")

    typer.echo(results_df.to_string())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@app.command()
def main(
    assess: bool = typer.Option(
        True, "--assess/--no-assess", help="Run wind + turbine assessments"
    ),
    force_save: bool = typer.Option(
        False, "--force-save/--no-force-save", help="Forces saving of all figures"
    ),
    do_comparison: bool = typer.Option(
        True, "--compare/--no-compare", help="Run turbine comparison"
    ),
    do_farm: bool = typer.Option(True, "--farm/--no-farm", help="Run microgrid comparison"),
) -> None:
    fin = MicrogridFinancialModel()
    wind_data = load_wind_resource()
    wind_resource = wind_data.create_wind_model()
    turbine_options = [load_nrel_6MW(), load_nrel_8MW(), load_nrel_10MW(), load_dtu_10MW()]

    # Ensure any figure saves can be forced
    global FORCE_SAVE
    FORCE_SAVE = force_save

    if assess:
        assess_extra()
        assess_wind_resource(wind_data)
        for turbine in turbine_options:
            assess_turbine(turbine, wind_resource)

    if do_comparison:
        best_turbine = compare_turbines(fin, wind_resource, turbine_options)
    else:
        # Fall back to lowest-LCOE without re-running if compare is skipped
        # — replace with a cached result or prompt if needed
        raise typer.Exit("Re-run with --compare to select a turbine before --farm.")

    if do_farm:
        options: dict[CO, WindFarm] = {}
        for config in CO:
            farm_dir = OUTPUT_DIR / "farm" / str(config)
            layout_path = farm_dir / "layout.json"

            typer.echo(f"── Wind farm assessment: {config}")
            if layout_path.exists() and not force_save:
                typer.echo("[skip] Loading preconfigured layout")
                farm = WindFarm.load_layout(layout_path, best_turbine)
            else:
                typer.echo("  Optimising layout...")

                optimiser = WindFarmOptimiser(config, best_turbine, wind_data)
                farm = optimiser.optimise(wind_resource)
                farm.save_layout(layout_path)

                typer.echo(
                    f"  Prevailing wind direction: {optimiser.prevailing_wind_direction():.1f}°"
                )
                typer.echo(f"  Optimized boundary angle:  {farm.angle_deg:.1f}°")

            _save(farm.plot_layout(wind_resource), farm_dir / "aep_per_turbine.jpeg")
            options[config] = farm

        compare_microgrids(fin, options, wind_resource)


if __name__ == "__main__":
    app()

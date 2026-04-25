from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pypsa

from ass1.modeling.farm import WindFarm
from ass1.modeling.location import WindResourceModel


class Microgrid:
    """
    Wind + battery microgrid for the Seaspray site.

    Uses PyPSA to solve a linear program that sizes the battery energy
    storage system (BESS) such that the hourly load is met at every
    timestep of the year. The objective is minimum cost, subject to:

        - Load must equal generation + battery discharge at every hour
        - Wind farm output is capped at the available (wake-adjusted) power
        - Battery state of charge must stay within 0 and its optimised capacity
        - Battery starts and ends the year at the same state of charge
          (cyclic constraint — reasonable for a 'typical year' analysis)

    A dummy slack generator is retained with a very high marginal cost so
    the LP stays feasible even if wind + battery cannot cover every hour.
    """

    BATTERY_EFFICIENCY: float = 0.90
    BATTERY_MAX_HOURS: float = 8.0
    SLACK_COST: float = 1_000_000.0

    def __init__(self, farm: WindFarm):
        self.farm = farm
        self.battery_energy_mwh: float | None = None
        self.battery_power_mw: float | None = None
        self.wind_cf: float | None = None
        self.slack_energy_mwh: float | None = None

        n = pypsa.Network()
        n.add("Bus", "Seaspray", carrier="electricity")
        n.add("Load", "demand", bus="Seaspray")
        n.add(
            "Generator",
            "dummy",
            bus="Seaspray",
            control="Slack",
            marginal_cost=self.SLACK_COST,
            p_nom=10_000,
        )
        n.add(
            "Generator",
            "wind_farm",
            bus="Seaspray",
            p_nom=self.farm.nameplate_capacity_mw,
            p_nom_extendable=False,
            p_min_pu=0.0,
            marginal_cost=0.0,
            carrier="wind",
        )
        n.add(
            "StorageUnit",
            "battery",
            bus="Seaspray",
            p_nom_extendable=True,
            max_hours=self.BATTERY_MAX_HOURS,
            efficiency_store=np.sqrt(self.BATTERY_EFFICIENCY),
            efficiency_dispatch=np.sqrt(self.BATTERY_EFFICIENCY),
            cyclic_state_of_charge=True,
            marginal_cost=0.0,
            capital_cost=0.0,
        )

        n.sanitize()
        self.n = n

    def prepare_network(self, site: pd.DataFrame, wind: WindResourceModel) -> None:
        """Set time-series inputs on the network."""
        site = _coerce_arrow_strings(site)
        ts = site["Time"]
        ws = site["WS50M"].to_numpy()
        wd = site["WD50M"].to_numpy()
        load_mw = site["Load [MW]"].to_numpy()

        self.n.set_snapshots(ts)
        self.n.loads_t.p_set["demand"] = load_mw

        windfarm_output_kw = self.farm.power_at_50m(wind, ws, wd)
        windfarm_output_mw = windfarm_output_kw / 1_000.0

        nameplate_mw = self.farm.nameplate_capacity_mw
        p_max_pu = windfarm_output_mw / nameplate_mw

        self.n.generators_t.p_max_pu["wind_farm"] = p_max_pu

        c_en_per_mwh = 300_000.0
        c_pow_per_mw = 375_000.0
        capital_cost_per_mw = max(
            c_en_per_mwh / self.BATTERY_MAX_HOURS,
            c_pow_per_mw,
        )
        self.n.storage_units.loc["battery", "capital_cost"] = capital_cost_per_mw

    def solve_network(self) -> tuple[float, float]:
        """Run the LP. Returns optimal battery energy capacity in MWh."""
        self.n.optimize()

        p_nom_opt_mw = float(self.n.storage_units.at["battery", "p_nom_opt"])
        e_nom_opt_mwh = p_nom_opt_mw * self.BATTERY_MAX_HOURS
        self.battery_power_mw = p_nom_opt_mw
        self.battery_energy_mwh = e_nom_opt_mwh

        wind_gen_mwh = float(self.n.generators_t.p["wind_farm"].sum())
        nameplate_annual_mwh = self.farm.nameplate_capacity_mw * 8760.0
        self.wind_cf = wind_gen_mwh / nameplate_annual_mwh if nameplate_annual_mwh else 0.0

        self.slack_energy_mwh = float(self.n.generators_t.p["dummy"].sum())

        return e_nom_opt_mwh, p_nom_opt_mw

    @property
    def total_load_mwh(self) -> float:
        return float(self.n.loads_t.p_set["demand"].sum())

    def plot_dispatch(self) -> go.Figure:
        """Stacked area chart: wind + battery discharge vs load over the year."""
        wind = self.n.generators_t.p["wind_farm"]
        slack = self.n.generators_t.p["dummy"]
        bat_dispatch = self.n.storage_units_t.p_dispatch["battery"]
        load = self.n.loads_t.p["demand"]

        w = 24 * 7
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=wind.index,
                y=wind.rolling(w, center=True).mean(),
                name="Wind (dispatched)",
                line=dict(color="#00d4ff", width=2),
                stackgroup="one",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bat_dispatch.index,
                y=bat_dispatch.rolling(w, center=True).mean(),
                name="Battery discharge",
                line=dict(color="#34d399", width=2),
                stackgroup="one",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=slack.index,
                y=slack.rolling(w, center=True).mean(),
                name="Slack (unmet)",
                line=dict(color="#f87171", width=2),
                stackgroup="one",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=load.index,
                y=load.rolling(w, center=True).mean(),
                name="Load",
                line=dict(color="#fbbf24", width=2.5, dash="dash"),
            )
        )
        fig.update_layout(
            title=dict(
                text=(
                    f"Dispatch – {self.farm.configuration.value} turbines "
                    f"({self.farm.nameplate_capacity_mw:.0f} MW) + "
                    f"{self.battery_energy_mwh:.0f} MWh battery"
                    "<br><sup>7-day rolling mean</sup>"
                ),
                x=0.5,
            ),
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Power (MW)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99),
        )
        return fig

    def plot_battery_soc(self) -> go.Figure:
        """Battery state of charge over the year."""
        soc = self.n.storage_units_t.state_of_charge["battery"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=soc.index,
                y=soc.values,
                mode="lines",
                line=dict(color="#a78bfa", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.15)",
                name="State of charge",
            )
        )
        fig.add_hline(
            y=self.battery_energy_mwh,
            line=dict(color="#f87171", width=1, dash="dash"),
            annotation_text=f"Capacity: {self.battery_energy_mwh:.0f} MWh",
        )
        fig.update_layout(
            title=dict(text="Battery state of charge", x=0.5),
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(
                title="Energy stored (MWh)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"
            ),
            hovermode="x unified",
        )
        return fig


def _coerce_arrow_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Arrow-backed string columns to numpy object dtype for PyPSA."""
    return df.apply(lambda col: col.astype(str) if hasattr(col.dtype, "pyarrow_dtype") else col)

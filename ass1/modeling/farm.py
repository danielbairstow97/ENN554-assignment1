from __future__ import annotations

import numpy as np
from enum import Enum
from py_wake import NOJ

from ass1.modeling.turbine import WindTurbine
from ass1.modeling.location import WindResourceModel, WindResourceData


class ConfigurationOption(Enum):
    NINE = 9
    SIXTEEN = 16
    TWENTY_FIVE = 25


class WindFarm:
    # Turbine chosen for the wind farm
    turbine: WindTurbine
    configuration: ConfigurationOption

    # 5D crosswind: minimizes land use while avoiding turbulence interference.
    # 7D downwind: allows wake to recover ~80% before hitting the next turbine.
    SPACING_CROSSWIND = 5  # multiples of rotor diameter, perpendicular to wind
    SPACING_DOWNWIND = 5  # multiples of rotor diameter, along wind direction

    def __init__(self, configuration: ConfigurationOption, turbine: WindTurbine, wind_data: WindResourceData):
        self.configuration = configuration
        self.turbine = turbine
        self.wind_data = wind_data

        # Build the WindFarm

        n = configuration.value  # actual number: 9, 16 or 25
        self.n_row = int(np.sqrt(n))  # grid side length: 3, 4 or 5
        D = turbine.diameter()  # rotor diameter

        self.spacing_x = self.SPACING_CROSSWIND * D  # ~891m between columns
        self.spacing_y = self.SPACING_DOWNWIND * D  # ~1248m between rows

        self.xy_grid = self._make_grid(staggered=False)
        self.xy_staggered = self._make_grid(staggered=True)

    def _make_grid(self, staggered: bool) -> tuple[np.ndarray, np.ndarray]:
        """ Generates x/y coordinates for a wind farm layout.
        staggered=False → regular grid (3x3, 4x4, 5x5)
        staggered=True  → every odd row is shifted by spacing_x/2
                          so turbines are not directly in each other's wake"""
        n = self.n_row
        xs, ys = [], []

        for row in range(n):
            # Shift every odd row by half the crosswind spacing
            offset = (self.spacing_x / 2) if (staggered and row % 2 == 1) else 0.0
            for col in range(n):
                xs.append(col * self.spacing_x + offset)
                ys.append(row * self.spacing_y)

        xs, ys = np.array(xs), np.array(ys)

        # Calculate prevailing wind direction using vector mean (circular average)
        # Simple mean is wrong for circular data — vector mean handles the 0°/360° boundary
        wd_rad = np.radians(np.array(self.wind_data.wind_direction))
        prevailing_wd = np.degrees(np.arctan2(np.mean(np.sin(wd_rad)),np.mean(np.cos(wd_rad)))) % 360
        # Rotate grid so rows are perpendicular to prevailing wind direction
        rot_angle = (prevailing_wd - 180) % 360
        angle_rad = np.radians(rot_angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = c * xs + s * ys
        y_rot = -s * xs + c * ys

        return x_rot, y_rot

    def power_at_50m(self,ws_50: np.ndarray,wd: np.ndarray,wind_model: WindResourceModel,) -> dict:
        """Calculate the output of the windfarm based on the input wind speed at 50m."""
        ws_50 = np.asarray(ws_50, dtype=float)
        wd = np.asarray(wd, dtype=float)

        # Scale wind speed from 50m to hub height using the power law (1/10 exponent)
        ws_hub = ws_50 * (self.turbine.hub_height() / 50.0) ** (1 / 10.0)

        site = wind_model.pywake_site
        model = NOJ(site, self.turbine)

        results = {}
        for layout_name, (x, y) in [("grid", self.xy_grid), ("staggered", self.xy_staggered)]:
            sim = model(x, y, ws=ws_hub, wd=wd)
            # Sum power over all turbines per hour, convert W → kW
            results[layout_name] = sim.Power.values.sum(axis=0) / 1_000

        return results

    @property
    def nameplate_capacity_mw(self) -> float:
        """Nameplate capacity of the wind farm in MW"""
        return self.configuration.value * self.turbine.rated_power_kw / 1_000

    def compare_layouts(self, wind_model: WindResourceModel) -> dict:
        """
        Compares grid vs. staggered layout by AEP using the WindResourceModel site.
        """
        site = wind_model.pywake_site
        model = NOJ(site, self.turbine)

        aep_grid = float(model(*self.xy_grid).aep().sum())
        aep_staggered = float(model(*self.xy_staggered).aep().sum())

        return {
            "n_turbines": self.configuration.value,
            "nameplate_MW": self.nameplate_capacity_mw,
            "aep_grid_GWh": round(aep_grid, 2),
            "aep_staggered_GWh": round(aep_staggered, 2),
            "best_layout": "staggered" if aep_staggered > aep_grid else "grid",
            "improvement_%": round(
                (aep_staggered - aep_grid) / aep_grid * 100, 2
            ),
        }

    def plot_aep_per_turbine(self, wind_model: WindResourceModel):
        """
        Plot AEP per turbine for both layouts side by side.
        The layout with higher total AEP is highlighted in the title.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        site = wind_model.pywake_site
        model = NOJ(site, self.turbine)

        layouts = {
            "grid": self.xy_grid,
            "staggered": self.xy_staggered,
        }

        # Calculate AEP per turbine for both layouts
        aep_per_layout = {}
        total_aep = {}
        for name, (x, y) in layouts.items():
            sim = model(x, y)
            # sum over wd and ws → one AEP value per turbine
            aep_per_turbine = sim.aep().sum(["wd", "ws"]).values
            aep_per_layout[name] = (x, y, aep_per_turbine)
            total_aep[name] = aep_per_turbine.sum()

        best = max(total_aep, key=total_aep.get)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Grid  —  AEP: {total_aep['grid']:.1f} GWh" + (" ✦ best" if best == "grid" else ""),
                f"Staggered  —  AEP: {total_aep['staggered']:.1f} GWh" + (" ✦ best" if best == "staggered" else ""),
            ],
        )

        for col, (name, (x, y, aep)) in enumerate(aep_per_layout.items(), start=1):
            is_best = name == best

            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode="markers+text",
                    text=[str(i) for i in range(len(x))],
                    textposition="top center",
                    textfont=dict(color="white", size=9),
                    marker=dict(
                        size=18,
                        color=aep,
                        colorscale="Plasma",
                        showscale=(col == 2),  # only show colorbar on right plot
                        colorbar=dict(
                            title="AEP [GWh]",
                            x=1.02,
                        ),
                        line=dict(
                            # highlight best layout with a bright border
                            color="white" if is_best else "rgba(0,0,0,0)",
                            width=2 if is_best else 0,
                        ),
                    ),
                    showlegend=False,
                ),
                row=1, col=col,
            )

        fig.update_layout(
            title=dict(
                text=f"AEP per Turbine — {self.configuration.value} turbines  |  Best: {best}",
                x=0.5,
            ),
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=520,
            width=1000,
        )
        fig.update_xaxes(title_text="x [m]", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(title_text="y [m]", showgrid=True, gridcolor="rgba(255,255,255,0.08)")

        return fig
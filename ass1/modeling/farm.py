from __future__ import annotations

from enum import IntEnum
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from scipy.optimize import minimize

from ass1.modeling.location import WindResourceData, WindResourceModel
from ass1.modeling.turbine import WindTurbine


class ConfigurationOption(IntEnum):
    NINE = 9
    SIXTEEN = 16
    TWENTY_FIVE = 25


class WindFarm:
    """
    Holds a fixed turbine layout and runs PyWake simulations against it.

    Construct directly with known positions, or use WindFarmOptimiser to
    derive an optimised layout and receive a WindFarm back.

    Persistence
    -----------
    save_layout(path)         — write layout to JSON
    WindFarm.load_layout(...) — reconstruct from JSON, skipping re-optimisation
    """

    BOUNDARY_SIZE_M: float = np.sqrt(3_000_000)  # ≈ 1732 m side length

    def __init__(
        self,
        configuration: ConfigurationOption,
        turbine: WindTurbine,
        x: np.ndarray,
        y: np.ndarray,
        angle_deg: float = 0.0,
        prevailing_wind: float = 0.0,
    ) -> None:
        self.configuration = configuration
        self.turbine = turbine
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.angle_deg = float(angle_deg)
        self.prevailing_wind = prevailing_wind

        if len(self.x) != configuration.value:
            raise ValueError(f"Expected {configuration.value} positions, got {len(self.x)}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nameplate_capacity_mw(self) -> float:
        return self.configuration.value * self.turbine.rated_power_kw / 1_000

    @property
    def n_turbines(self) -> int:
        return self.configuration.value

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _wf_model(self, wind_model: WindResourceModel) -> Bastankhah_PorteAgel_2014:
        return Bastankhah_PorteAgel_2014(wind_model.pywake_site, self.turbine, k=0.04)

    def simulate(self, wind_model: WindResourceModel):
        """Run a full PyWake simulation and return the SimulationResult."""
        return self._wf_model(wind_model)(self.x, self.y)

    def aep(self, wind_model: WindResourceModel) -> float:
        """Wake-affected farm AEP (GWh/year)."""
        return float(self.simulate(wind_model).aep().sum())

    def power_at_50m(
        self,
        wind_model: WindResourceModel,
        ws_50: np.ndarray,
        wd: np.ndarray,
    ) -> np.ndarray:
        """
        Farm power output (kW) per timestep for a time series of wind
        speed and direction measured at 50 m reference height.
        """
        print("Adjusting wind")
        ws_hub = self.turbine.ws_at_hub(ws_50)

        ws_hub = np.asarray(ws_hub, dtype=float)
        wd = np.asarray(wd, dtype=float)

        model = self._wf_model(wind_model)
        sim = model(self.x, self.y, ws=ws_hub, wd=wd, time=True)

        return sim.Power.values.sum(axis=0) / 1_000  # W → kW

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_layout(self, path: str | Path) -> None:
        """
        Save the turbine layout to a JSON file.

        Stored fields
        -------------
        configuration : ConfigurationOption name  (e.g. "SIXTEEN")
        turbine_name  : str  — cross-checked on load
        angle_deg     : float — boundary rotation
        x, y          : list[float] — turbine easting / northing (m)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "configuration": self.configuration.name,
            "turbine_name": self.turbine.name_str,
            "angle_deg": self.angle_deg,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"[WindFarm] Layout saved → {path}")

    @classmethod
    def load_layout(
        cls,
        path: str | Path,
        turbine: WindTurbine,
    ) -> "WindFarm":
        """
        Reconstruct a WindFarm from a JSON file written by save_layout().

        Parameters
        ----------
        path    : path to the saved JSON file
        turbine : WindTurbine to attach — name is checked against the file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Layout file not found: {path}")

        data = json.loads(path.read_text())

        if data["turbine_name"] != turbine.name_str:
            import warnings

            warnings.warn(
                f"Layout was optimised for '{data['turbine_name']}' "
                f"but loading with '{turbine.name_str}'. Results may differ.",
                stacklevel=2,
            )

        farm = cls(
            configuration=ConfigurationOption[data["configuration"]],
            turbine=turbine,
            x=np.array(data["x"]),
            y=np.array(data["y"]),
            angle_deg=float(data["angle_deg"]),
        )
        print(f"[WindFarm] Layout loaded ← {path}")
        return farm

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def boundary(self) -> np.ndarray:
        """Return the (4, 2) rotated boundary corners."""
        size = self.BOUNDARY_SIZE_M
        corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=float)
        rad = np.radians(self.angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        return (np.array([[c, -s], [s, c]]) @ corners.T).T

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_layout(self, wind_model: WindResourceModel) -> go.Figure:
        """
        Turbine positions coloured by individual AEP, rotated boundary
        outline, and a prevailing wind direction arrow.
        """
        sim = self.simulate(wind_model)
        aep_per_turbine = sim.aep().sum(["wd", "ws"]).values
        total_aep = aep_per_turbine.sum()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.x,
                y=self.y,
                mode="markers+text",
                text=[str(i) for i in range(self.n_turbines)],
                textposition="top center",
                textfont=dict(color="white", size=9),
                marker=dict(
                    size=18,
                    color=aep_per_turbine,
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="AEP [GWh]"),
                    line=dict(color="white", width=1),
                ),
                showlegend=False,
            )
        )

        bnd = np.vstack([self.boundary(), self.boundary()[0]])
        fig.add_trace(
            go.Scatter(
                x=bnd[:, 0],
                y=bnd[:, 1],
                mode="lines",
                line=dict(color="white", width=1.5, dash="dash"),
                showlegend=False,
            )
        )

        cx, cy = self.boundary()[:, 0].mean(), self.boundary()[:, 1].mean()
        length = self.BOUNDARY_SIZE_M * 0.2
        rad = np.radians(self.prevailing_wind)
        dx, dy = -length * np.sin(rad), -length * np.cos(rad)

        fig.add_annotation(
            x=cx + dx,
            y=cy + dy,
            ax=cx - dx,
            ay=cy - dy,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#00d4ff",
            text=f"Wind {self.prevailing_wind:.0f}°",
            font=dict(color="#00d4ff", size=11),
        )

        fig.update_layout(
            title=dict(
                text=f"Optimised Layout — {self.n_turbines} turbines  |  "
                f"Total AEP: {total_aep:.1f} GWh",
                x=0.5,
            ),
            xaxis=dict(
                title="x [m]",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                scaleanchor="y",
                range=[0, self.BOUNDARY_SIZE_M],
            ),
            yaxis=dict(
                title="y [m]",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                range=[0, self.BOUNDARY_SIZE_M],
            ),
            height=600,
            width=700,
        )
        return fig


class WindFarmOptimiser:
    """
    Optimises turbine positions and boundary rotation for a given
    configuration, turbine and wind resource, then returns a WindFarm.

    Usage
    -----
    optimiser = WindFarmOptimiser(ConfigurationOption.SIXTEEN, turbine, wind_data)
    farm      = optimiser.optimise(wind_model)
    farm.save_layout("outputs/layouts/farm_16.json")

    # Later run — skip optimisation entirely:
    farm = WindFarm.load_layout("outputs/layouts/farm_16.json", turbine)
    """

    def __init__(
        self,
        configuration: ConfigurationOption,
        turbine: WindTurbine,
        wind_data: WindResourceData,
    ) -> None:
        self.configuration = configuration
        self.turbine = turbine
        self.wind_data = wind_data

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def optimise(self, wind_model: WindResourceModel) -> WindFarm:
        """
        Run Smart Start initialisation then SLSQP layout optimisation.
        Returns a WindFarm ready for simulation and persistence.
        """
        print(f"[Optimiser] Smart Start — {self.configuration.value} turbines")
        x_init, y_init = self._smart_start(wind_model)

        print("[Optimiser] SLSQP optimisation")
        x_opt, y_opt, angle_opt = self._slsqp(wind_model, x_init, y_init)

        return WindFarm(
            configuration=self.configuration,
            turbine=self.turbine,
            x=x_opt,
            y=y_opt,
            angle_deg=angle_opt,
            prevailing_wind=self.prevailing_wind_direction(),
        )

    # ------------------------------------------------------------------
    # Smart Start initialisation
    # ------------------------------------------------------------------

    def prevailing_wind_direction(self) -> float:
        wd_rad = np.radians(np.array(self.wind_data.wind_direction))
        return float(
            np.degrees(np.arctan2(np.mean(np.sin(wd_rad)), np.mean(np.cos(wd_rad)))) % 360
        )

    def _smart_start(self, wind_model: WindResourceModel) -> tuple[np.ndarray, np.ndarray]:
        """
        Greedy turbine placement: add one turbine at a time at the
        candidate position that maximises incremental AEP.
        Minimum spacing relaxes progressively if no valid candidates exist.
        """
        wfm = Bastankhah_PorteAgel_2014(wind_model.pywake_site, self.turbine, k=0.04)
        n_wt = self.configuration.value
        D = self.turbine.diameter()
        size = WindFarm.BOUNDARY_SIZE_M

        g = np.linspace(size * 0.05, size * 0.95, 30)
        xx, yy = np.meshgrid(g, g)
        x_cand, y_cand = xx.flatten(), yy.flatten()

        placed_x: list[float] = []
        placed_y: list[float] = []

        for i in range(n_wt):
            valid: list[tuple[float, float]] = []

            for min_spacing in [2 * D, 1.5 * D, 1.0 * D, 0.5 * D]:
                for cx, cy in zip(x_cand, y_cand):
                    if not any(
                        np.hypot(cx - px, cy - py) < min_spacing
                        for px, py in zip(placed_x, placed_y)
                    ):
                        valid.append((cx, cy))
                if valid:
                    break

            if not valid:
                best_x, best_y = max(
                    zip(x_cand, y_cand),
                    key=lambda p: (
                        min(np.hypot(p[0] - px, p[1] - py) for px, py in zip(placed_x, placed_y))
                        if placed_x
                        else 0.0
                    ),
                )
            else:
                best_aep = -np.inf
                best_x, best_y = valid[0]
                for cx, cy in valid:
                    aep = float(
                        wfm(np.array(placed_x + [cx]), np.array(placed_y + [cy])).aep().sum()
                    )
                    if aep > best_aep:
                        best_aep = aep
                        best_x, best_y = cx, cy

            placed_x.append(best_x)
            placed_y.append(best_y)
            print(f"  Turbine {i + 1}/{n_wt} → ({best_x:.0f}, {best_y:.0f})")

        return np.array(placed_x), np.array(placed_y)

    # ------------------------------------------------------------------
    # SLSQP optimisation
    # ------------------------------------------------------------------

    def _slsqp(
        self,
        wind_model: WindResourceModel,
        x_init: np.ndarray,
        y_init: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Jointly optimise x/y turbine positions and boundary rotation angle.
        Design vector: [x₀…xₙ, y₀…yₙ, angle]
        """
        wfm = Bastankhah_PorteAgel_2014(wind_model.pywake_site, self.turbine, k=0.04)
        n_wt = self.configuration.value
        D = self.turbine.diameter()

        xy0 = np.concatenate([x_init, y_init, [0.0]])

        def neg_aep(xy: np.ndarray) -> float:
            return -float(wfm(xy[:n_wt], xy[n_wt : 2 * n_wt]).aep().sum())

        def _corners(angle_deg: float) -> np.ndarray:
            size = WindFarm.BOUNDARY_SIZE_M
            corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=float)
            rad = np.radians(angle_deg)
            c, s = np.cos(rad), np.sin(rad)
            return (np.array([[c, -s], [s, c]]) @ corners.T).T

        def _half_plane(p1: np.ndarray, p2: np.ndarray, xy: np.ndarray) -> float:
            x, y = xy[:n_wt], xy[n_wt : 2 * n_wt]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            return float(((x - p1[0]) * normal[0] + (y - p1[1]) * normal[1]).min())

        def boundary_constraints(xy: np.ndarray) -> list[float]:
            bnd = _corners(xy[2 * n_wt])
            return [_half_plane(bnd[i], bnd[(i + 1) % 4], xy) for i in range(4)]

        def min_spacing(xy: np.ndarray) -> np.ndarray:
            x, y = xy[:n_wt], xy[n_wt : 2 * n_wt]
            return np.array(
                [
                    np.hypot(x[i] - x[j], y[i] - y[j]) - 2 * D
                    for i in range(n_wt)
                    for j in range(i + 1, n_wt)
                ]
            )

        constraints = [
            {"type": "ineq", "fun": lambda xy, i=i: boundary_constraints(xy)[i]} for i in range(4)
        ] + [{"type": "ineq", "fun": min_spacing}]
        bounds = [(None, None)] * n_wt + [(None, None)] * n_wt + [(0, 90)]

        result = minimize(
            fun=neg_aep,
            x0=xy0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-2, "disp": True},
        )

        x_opt, y_opt = result.x[:n_wt], result.x[n_wt : 2 * n_wt]
        angle_opt = result.x[2 * n_wt]
        print(f"[Optimiser] AEP: {-result.fun:.2f} GWh  |  angle: {angle_opt:.1f}°")
        return x_opt, y_opt, angle_opt

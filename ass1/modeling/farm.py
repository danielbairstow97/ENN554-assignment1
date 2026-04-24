from __future__ import annotations

import numpy as np
from enum import Enum
from py_wake import NOJ
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from scipy.optimize import minimize

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

    BOUNDARY_SIZE_M = np.sqrt(3_000_000)

    def __init__(self, configuration: ConfigurationOption, turbine: WindTurbine, wind_data: WindResourceData):
        self.configuration = configuration
        self.turbine = turbine
        self.wind_data = wind_data

        # xy_optimized is set by optimize_layout() — None until then
        self.xy_optimized: tuple[np.ndarray, np.ndarray] | None = None
        self._optimized_angle: float | None = None

    def _prevailing_wind_direction(self) -> float:
        """
        Calculate prevailing wind direction using vector mean (circular average).
        Simple mean is wrong for circular data — vector mean handles 0°/360° boundary.
        """
        wd_rad = np.radians(np.array(self.wind_data.wind_direction))
        return float(np.degrees(np.arctan2(np.mean(np.sin(wd_rad)), np.mean(np.cos(wd_rad)))) % 360)

    def _make_boundary(self, angle_deg: float | None = None) -> np.ndarray:
        """
        Build the 3km² square boundary rotated by angle_deg.
        If angle_deg is None, uses prevailing wind direction.
        After optimize_layout(), uses the optimized angle automatically.
        """
        size = self.BOUNDARY_SIZE_M

        # Use optimized angle if available, otherwise prevailing wind direction
        if angle_deg is None:
            angle_deg = getattr(self, '_optimized_angle', 0.0)

        corners = np.array([
            [0, 0],
            [size, 0],
            [size, size],
            [0, size],
        ], dtype=float)

        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array([[c, -s], [s, c]])

        return (rot_matrix @ corners.T).T

    def _make_start_pos(self, wind_model: WindResourceModel) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate starting positions using a Smart Start inspired approach.
        Places turbines one by one at the position that maximises AEP.
        Automatically relaxes minimum spacing if not enough candidates available.
        """
        site = wind_model.pywake_site
        wfm = Bastankhah_PorteAgel_2014(site, self.turbine, k=0.04)
        n_wt = self.configuration.value
        D = self.turbine.diameter()
        size = self.BOUNDARY_SIZE_M

        # Denser candidate grid — more options for tight configurations
        n_cand = 30  # 30×30 = 900 candidates
        x_cand = np.linspace(size * 0.05, size * 0.95, n_cand)
        y_cand = np.linspace(size * 0.05, size * 0.95, n_cand)
        xx, yy = np.meshgrid(x_cand, y_cand)
        x_cand = xx.flatten()
        y_cand = yy.flatten()

        placed_x = []
        placed_y = []

        for i in range(n_wt):

            # Start with 2D minimum spacing, relax if no valid candidates found
            for min_spacing in [2 * D, 1.5 * D, 1 * D, 0.5 * D]:
                valid_candidates = []

                for cx, cy in zip(x_cand, y_cand):
                    too_close = any(
                        np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < min_spacing
                        for px, py in zip(placed_x, placed_y)
                    )
                    if not too_close:
                        valid_candidates.append((cx, cy))

                if valid_candidates:
                    break  # found valid candidates at this spacing — use them

            if not valid_candidates:
                # Absolute fallback — just pick the candidate furthest from all placed turbines
                best_x, best_y = max(
                    zip(x_cand, y_cand),
                    key=lambda p: min(
                        np.sqrt((p[0] - px) ** 2 + (p[1] - py) ** 2)
                        for px, py in zip(placed_x, placed_y)
                    ) if placed_x else 0
                )
            else:
                # Find best AEP among valid candidates
                best_aep = -np.inf
                best_x = valid_candidates[0][0]
                best_y = valid_candidates[0][1]

                for cx, cy in valid_candidates:
                    xs = np.array(placed_x + [cx])
                    ys = np.array(placed_y + [cy])
                    aep = float(wfm(xs, ys).aep().sum())

                    if aep > best_aep:
                        best_aep = aep
                        best_x = cx
                        best_y = cy

            placed_x.append(best_x)
            placed_y.append(best_y)
            print(f"  Smart Start: placed turbine {i + 1}/{n_wt} "
                  f"at ({best_x:.0f}, {best_y:.0f}) "
                  f"[min spacing used: {min_spacing / D:.1f}D]")

        return np.array(placed_x), np.array(placed_y)

    def optimize_layout(self, wind_model: WindResourceModel) -> tuple[np.ndarray, np.ndarray]:
        """
        Optimise turbine positions AND boundary rotation within the 3km² square.

        Design variables:
            - x, y positions of all turbines
            - angle: rotation of the boundary (±90° around prevailing wind direction)
              A square has 4-fold symmetry so ±90° covers all unique orientations.

        Objective: maximise AEP (minimise negative AEP)

        Constraint:
            - All turbines must stay inside the rotated 1732m × 1732m square
            - Boundary is rebuilt at each iteration using the current angle
        """
        site = wind_model.pywake_site
        wfm = Bastankhah_PorteAgel_2014(site, self.turbine, k=0.04)
        n_wt = self.configuration.value

        # Starting positions — evenly spread inside the boundary
        x_init, y_init = self._make_start_pos(wind_model)
        angle_init = 0.0

        # Flatten into single array: [x0..xn, y0..yn, angle]
        xy_init = np.concatenate([x_init, y_init, [angle_init]])

        # --- Objective function ---
        def neg_aep(xy):
            x = xy[:n_wt]
            y = xy[n_wt:2 * n_wt]
            aep = float(wfm(x, y).aep().sum())
            print(f"  AEP: {aep:.2f} GWh")
            return -aep

        # --- Boundary builder ---
        def get_boundary(angle_deg: float) -> np.ndarray:
            """
            Build the 1732m × 1732m square boundary rotated by angle_deg.
            Returns shape (4, 2) corner coordinates.
            """
            size = self.BOUNDARY_SIZE_M
            corners = np.array([
                [0, 0],
                [size, 0],
                [size, size],
                [0, size],
            ], dtype=float)
            angle_rad = np.radians(angle_deg)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = np.array([[c, -s], [s, c]])
            return (rot_matrix @ corners.T).T

        # --- Boundary constraint ---
        def make_half_plane(p1: np.ndarray, p2: np.ndarray, xy: np.ndarray) -> float:
            """
            Signed distance of all turbines from edge p1→p2.
            Returns minimum distance — must be >= 0 for all turbines to be inside.
            """
            x = xy[:n_wt]
            y = xy[n_wt:2 * n_wt]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            return float(((x - p1[0]) * normal[0] + (y - p1[1]) * normal[1]).min())

        def boundary_constraints(xy: np.ndarray) -> list:
            """Rebuild boundary at current angle and return 4 half-plane constraints."""
            angle = xy[2 * n_wt]  # last element is the rotation angle
            boundary = get_boundary(angle)
            return [
                make_half_plane(boundary[i], boundary[(i + 1) % 4], xy)
                for i in range(4)
            ]

        # One SciPy constraint per boundary edge — all must be >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda xy, i=i: boundary_constraints(xy)[i]}
            for i in range(4)
        ]

        def min_spacing(xy: np.ndarray) -> np.ndarray:
            """
            Calculates pairwise distances between all turbines minus 2D.
            SciPy requires all values >= 0 — so every pair must be at least 2D apart.
            """
            x = xy[:n_wt]
            y = xy[n_wt:2 * n_wt]
            D = self.turbine.diameter()  # 178.3m → 2D = 356.6m

            distances = []
            for i in range(n_wt):
                for j in range(i + 1, n_wt):
                    # Euclidean distance between turbine i and turbine j
                    dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                    # subtract 2D — result must be >= 0
                    # e.g. dist=400m → 400-356=44  ✅ far enough
                    # e.g. dist=100m → 100-356=-256 ❌ too close
                    distances.append(dist - 2 * D)

            return np.array(distances)

        constraints.append({
            'type': 'ineq',
            'fun': min_spacing,
        })
        # --- Angle bounds ---
        # Turbine positions: unbounded (boundary constraint handles this)
        # Angle: ±90° around initial angle — covers all unique square orientations
        bounds = (
                [(None, None)] * n_wt +  # x positions
                [(None, None)] * n_wt +  # y positions
                [(0, 90)]  # rotation angle
        )

        # --- Run optimisation ---
        result = minimize(fun=neg_aep,x0=xy_init,method='SLSQP',constraints=constraints,bounds=bounds,options={'maxiter': 500,'ftol': 1e-5,'disp': True,},)

        # Extract optimised positions and angle
        x_opt = result.x[:n_wt]
        y_opt = result.x[n_wt:2 * n_wt]
        angle_opt = result.x[2 * n_wt]

        # Store optimised angle so plot_aep_per_turbine can draw the correct boundary
        self._optimized_angle = angle_opt

        self.xy_optimized = (x_opt, y_opt)
        print(f"  AEP:           {-result.fun:.2f} GWh")
        return self.xy_optimized

    def power_at_50m(self,ws_50: np.ndarray,wd: np.ndarray,wind_model: WindResourceModel,) -> np.ndarray:
        """
        Calculate farm power output (kW) per hour using the optimised layout.
        optimize_layout() must be called before this method.
        """
        if self.xy_optimized is None:
            raise RuntimeError("Call optimize_layout() before power_at_50m()")

        ws_50 = np.asarray(ws_50, dtype=float)
        wd    = np.asarray(wd,    dtype=float)

        # Scale wind speed from 50m to hub height using the power law (1/7 exponent)
        ws_hub = ws_50 * (self.turbine.hub_height() / 50.0) ** (1 / 7.0)

        site  = wind_model.pywake_site
        model = Bastankhah_PorteAgel_2014(site, self.turbine, k=0.04)

        x, y = self.xy_optimized
        sim  = model(x, y, ws=ws_hub, wd=wd)

        # Sum power over all turbines per hour, W → kW
        return sim.Power.values.sum(axis=0) / 1_000

    @property
    def nameplate_capacity_mw(self) -> float:
        """Nameplate capacity of the wind farm in MW"""
        return self.configuration.value * self.turbine.rated_power_kw / 1_000

    def plot_aep_per_turbine(self, wind_model: WindResourceModel):
        """
        Plot AEP per turbine for the optimised layout.
        optimize_layout() must be called before this method.
        Color indicates AEP — brighter turbines receive more wind (less wake).
        """
        import plotly.graph_objects as go

        if self.xy_optimized is None:
            raise RuntimeError("Call optimize_layout() before plot_aep_per_turbine()")

        site  = wind_model.pywake_site
        model = Bastankhah_PorteAgel_2014(site, self.turbine, k=0.04)

        x, y            = self.xy_optimized
        sim             = model(x, y)
        aep_per_turbine = sim.aep().sum(["wd", "ws"]).values
        total_aep       = aep_per_turbine.sum()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="markers+text",
                text=[str(i) for i in range(len(x))],
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

        boundary = self._make_boundary()
        # Close the square by appending the first corner at the end
        boundary_closed = np.vstack([boundary, boundary[0]])

        fig.add_trace(
            go.Scatter(
                x=boundary_closed[:, 0],
                y=boundary_closed[:, 1],
                mode="lines",
                line=dict(color="white", width=1.5, dash="dash"),
                showlegend=False,  # ← no legend
            )
        )

        center_x = boundary[:, 0].mean()
        center_y = boundary[:, 1].mean()
        arrow_length = self.BOUNDARY_SIZE_M * 0.2  # 20% of boundary size

        wd_rad = np.radians(self._prevailing_wind_direction())
        # Wind direction is where wind comes FROM — arrow shows where it goes TO
        dx = -arrow_length * np.sin(wd_rad)
        dy = -arrow_length * np.cos(wd_rad)

        fig.add_annotation(
            x=center_x + dx,  # arrow tip
            y=center_y + dy,
            ax=center_x - dx,  # arrow tail
            ay=center_y - dy,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#00d4ff",
            text=f"Wind {self._prevailing_wind_direction():.0f}°",
            font=dict(color="#00d4ff", size=11),
        )

        fig.update_layout(
            title=dict(
                text=(
                    f"Optimised Layout — {self.configuration.value} turbines"
                    f"  |  Total AEP: {total_aep:.1f} GWh"
                ),
                x=0.5,
            ),
            xaxis=dict(title="x [m]", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="y [m]", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=600,
            width=700,
        )

        return fig
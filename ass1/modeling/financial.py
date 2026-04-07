from dataclasses import dataclass

from ass1.modeling.farm import WindFarm
from ass1.modeling.turbine import WindTurbine


@dataclass
class FinancialModel:
    wt_rotor_cost: float = 170  # $/m^2
    wt_tower_cost: float = 90  # $/m^2
    wt_other_cost: float = 1000  # $/kW
    wt_foc: float = 60  # $/kW

    bess_energy_cost: float = 300  # $/kWh
    bess_power_cost: float = 375  # $/kW
    bess_foc: float = 10  # $/kW

    project_lifetime = 25  # years
    fcr: float = 0.05

    # @property
    # def fcr(self):

    def cost_turbine(self, turbine: WindTurbine) -> tuple[float, dict[str, float]]:
        costs = {}
        costs["Rotor"] = self.wt_rotor_cost * turbine.rotor_area
        costs["Tower"] = self.wt_tower_cost * turbine.rotor_area
        costs["Other"] = self.wt_other_cost * turbine.rated_power_kw

        return sum(costs.values()), costs

    def cost_turbine_opex(self, turbine):
        return self.wt_foc * turbine.rated_power_kw

    def lcoe(self, turbine: WindTurbine, aep: float) -> float:
        capex, _ = self.cost_turbine(turbine)
        opex = self.cost_turbine_opex(turbine)

        ucrf = self.fcr * (1 + self.fcr) ** self.project_lifetime / ((1 + self.fcr) - 1)

        lcoe = (ucrf * capex + opex) / (aep)

        return lcoe

    def cost_windfarm(self, farm: WindFarm):
        return 0.0

from dataclasses import dataclass


@dataclass
class FinancialModel:
    wt_rotor_cost: float = 170  # $/m^2
    wt_tower_cost: float = 90  # $/m^2
    wt_other_cost: float = 10000  # $/m^2
    wt_foc: float = 60  # $/kW

    bess_energy_cost: float = 300  # $/kWh
    bess_power_cost: float = 375  # $/kW
    bess_foc: float = 10  # $/kW

    fcr: float = 0.05

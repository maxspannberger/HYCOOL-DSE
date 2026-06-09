class Component:
    # default parameters for all components
    def __init__(self, name, trl):
        self.name = name
        self.trl = trl

class PowerComponent(Component):
    # class for power providing components
    def __init__(self, name, power_density, efficiency, trl, power_density_std=0.0, efficiency_std=0.0, volumetric_density=0.0):
        super().__init__(name, trl)
        self.power_density = power_density
        self.efficiency = efficiency / 100 if efficiency else None
        self.power_density_std = power_density_std
        self.efficiency_std = efficiency_std / 100
        self.volumetric_density = volumetric_density

class StorageComponent(Component):
    # class for energy storage components (batteries)
    def __init__(self, name, energy_density, power_density, efficiency, trl, energy_density_std=0.0, power_density_std=0.0, efficiency_std=0.0):
        super().__init__(name, trl)
        self.energy_density = energy_density
        self.power_density = power_density
        self.efficiency = efficiency / 100 if efficiency else None
        self.energy_density_std = energy_density_std
        self.power_density_std = power_density_std
        self.efficiency_std = efficiency_std / 100

class PipingComponent(Component):
    # class for pipes
    def __init__(self, name, heat_flux, mass_per_length, trl):
        super().__init__(name, trl)
        self.heat_flux = heat_flux
        self.mass_per_length = mass_per_length

class CableComponent(Component):
    # class for cables
    def __init__(self, name, power_density, mass_per_length, trl):
        super().__init__(name, trl)
        self.power_density = power_density
        self.mass_per_length = mass_per_length
        
class HeatExchangeComponent(Component):
    # class for HEX
    def __init__(self, name, mass_increase, efficiency_increase, trl):
        super().__init__(name, trl)
        self.mass_increase = mass_increase
        self.efficiency_increase = efficiency_increase

# =============================================================================
# Define a dictionary containing the data
# =============================================================================
# TODO: adjust standard deviations
component_params = {
    # Power Components
    "gt": PowerComponent("Gas Turbine", 10, 35, 5, power_density_std=1, efficiency_std=9),
    "fc_with_hex": PowerComponent("Fuel Cell", 2.83, 51, 3, power_density_std=0.283, efficiency_std=7),
    "hts_gen": PowerComponent("HTS Motor", 36, 99.9, 4, power_density_std=9),
    "hts_pow": PowerComponent("HTS Motor", 36, 99.9, 4, power_density_std=9),
    "gt_hex": PowerComponent("Gas Turbine + HEX ", 9.62, 39.5, 3, power_density_std=0.962, efficiency_std=9),
    "dc_dc_1": PowerComponent("DC-DC Converter 1", 17, 99.47, 3),
    "dc_dc_2": PowerComponent("DC-DC Converter 2", 17, 99.47, 3),
    "ac_dc": PowerComponent("AC-DC Rectifier", 35.1, 98.9, 3, volumetric_density=70000),
    "dc_ac": PowerComponent("DC-AC Inverter", 35.1, 98.9, 3, volumetric_density=70000),
    "bus": PowerComponent("Electric Bus", 26.8, 99.4, 3, volumetric_density=70000),

    # Battery
    "bt": StorageComponent("Battery", 0.510, 1.53, 90, 3, energy_density_std=0.102, power_density_std=0.306, efficiency_std=1),

    # Pipes
    "pipe": PipingComponent("Pipe", 9.89, 1.801, 6),

    # Cables
    "cable": CableComponent("Cable", 25, 8.112, 4),

    # Heat Exchangers
    "hex_gt": HeatExchangeComponent("HEX for Gas Turbine", 4, 4.5, 3),
    "hex_fc": HeatExchangeComponent("HEX for Fuel Cell", 35, 95, 4)
}
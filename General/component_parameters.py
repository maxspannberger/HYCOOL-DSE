class Component:
    # default parameters for all components
    def __init__(self, name, trl):
        self.name = name
        self.trl = trl

class PowerComponent(Component):
    # class for power providing components
    def __init__(self, name, power_density, efficiency, trl):
        super().__init__(name, trl)
        self.power_density = power_density
        self.efficiency = efficiency / 100 if efficiency else None

class StorageComponent(Component):
    # class for energy storage components (batteries)
    def __init__(self, name, energy_density, power_density, efficiency, trl):
        super().__init__(name, trl)
        self.energy_density = energy_density
        self.power_density = power_density
        self.efficiency = efficiency / 100 if efficiency else None

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
component_params = {
    # Power Components
    "gt": PowerComponent("Gas Turbine", 10, 35, 5),
    "fc_with_hex": PowerComponent("Fuel Cell", 2.83, 51, 3),
    "hts_gen": PowerComponent("HTS Motor", 20, 99.9, 4),
    "hts_pow": PowerComponent("HTS Motor", 20, 99.9, 4),
    "gt_hex": PowerComponent("Gas Turbine + HEX ", 9.62, 39.5, 3),
    "dc_dc_1": PowerComponent("DC-DC Converter 1", 17, 99.47, 3),
    "dc_dc_2": PowerComponent("DC-DC Converter 2", 17, 99.47, 3),
    "ac_dc": PowerComponent("AC-DC Rectifier", 21.1, 98.9, 3),
    "dc_ac": PowerComponent("DC-AC Inverter", 21.1, 98.9, 3),

    # Battery
    "bt": StorageComponent("Battery", 0.510,1.53, 90, 3),

    # Pipes
    "pipe": PipingComponent("Pipe", 9.89, 1.801, 6),

    # Cables
    "cable": CableComponent("Cable", 25, 8.112, 4),

    # Heat Exchangers
    "hex_gt": HeatExchangeComponent("HEX for Gas Turbine", 4, 4.5, 3),
    "hex_fc": HeatExchangeComponent("HEX for Fuel Cell", 35, 95, 4)
}
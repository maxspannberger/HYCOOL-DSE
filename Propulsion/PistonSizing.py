from TurbineSizing import GasTurbineCycle
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns
from rich.panel import Panel
from rich import box
from CoolProp.CoolProp import PropsSI

engine = GasTurbineCycle()
engine.size()

results = engine.results
design  = engine._design

_console = Console()

fluid   = 'ParaHydrogen'

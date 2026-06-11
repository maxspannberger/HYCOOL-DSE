import numpy as np
from dataclasses import dataclass, replace
import sys
from pathlib import Path
import json

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from WeightEstimations.mainClassII   import run_class_ii
from WeightEstimations.Aircraft_Config   import AircraftConfig, default_q400_hycool
from General.component_parameters import component_params as comp_params

cfg = default_q400_hycool()
result1 = run_class_ii(cfg,comp=comp_params, tol=1.0, max_iter=100, verbose=True)
ground_clearance_needed = 2/3*2.5+0.18+0.3 # 2.5 m for open fan diameter + 0.18 m for safety margin +0.3 m for wing box underneath fuselage
print(f"Ground clearance needed for the propellers: {ground_clearance_needed:.2f} m")
height_inside_dihedral=np.tan(11*np.pi/180)*result1.Wing_span/2*0.35
print(f"Clearance of the wing at 35% span with 11 degree dihedral: {height_inside_dihedral:.2f} m")
i=0.35
clearance_status = "insufficient"
iterations=0
while clearance_status == "insufficient" and i<1:
        i=i+0.01
        print(i)
        height_outside_dihedral=np.tan(7*np.pi/180)*(i*result1.Wing_span/2-result1.Wing_span/2*0.35)
        actual_clearance=height_inside_dihedral+height_outside_dihedral
        # print(f"Clearance of the wing at {i:.2f} span with {i:.2f} degree dihedral: {actual_clearance:.2f} m")
        clearance=actual_clearance-ground_clearance_needed
        print(f"Clearance margin at {i:.2f} span: {clearance:.2f} m")
        clearance_status = "sufficient" if actual_clearance >= ground_clearance_needed else "insufficient"
        iterations+=1

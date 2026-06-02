from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from h2_components import Tank, Pipe

'''
The wall of the components Pipe and Tank should be defined in the following manner.
 
- Wall contains a list of tupples. 
- Within the tupple the material is specified at index 0 and the thickness
  at index 1. 
- The materials should be oredered from inner tube to outer tube

For example: wall = [('ss-316l', 0.01), ('polyurethene', 0.02)]
eg. the inner layer is ss-326l and the outer layer is polyurethene

'''

wall = [('ss-316l',      0.01), 
        ('polyurethene', 0.02)]

system = [
    Tank(diameter=0.1, wall=wall), 
    Pipe(position=1, length=1.0, diameter=0.1, wall=wall, segments=10)
    ]


          


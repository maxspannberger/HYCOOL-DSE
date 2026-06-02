from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.append(str(root))

import insulation
import numpy as np



class H2Component:
    # default parameters for all components
    def __init__(self, name: str, position: int):
        self.name = name
        self.position = position

class Pipe(H2Component):
    def __init__(self, 
                 name:str, 
                 position: int, 
                 length: float, 
                 diameter: float, 
                 wall: list,
                 segment_length: float = 0.01):
        
        super().__init__(name, position)
        self.N = len(wall)
        
class HEX(H2Component):
    def __init__(self,
                 name:str, 
                 position: int,
                 catalyst: bool = False):
    
        super().__init__(name, position)
    
        

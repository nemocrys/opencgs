from dataclasses import dataclass

@dataclass
class HeatfluxSurf:  # surface with heat flux
    ID: int
    BodyIDs: list
    lmbd: float

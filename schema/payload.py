from pydantic import BaseModel
from typing import Literal

class GnenrateRequest(BaseModel):
    pickup_hr : int
    pickup_min : int
    dropoff_hr : int
    dropoff_min : int
    passengers : int
    distance : float
    fare : float
    tip : float
    tolls : float
    color : Literal['yellow','green']
    payment : Literal['credit card','cash']
    pickup_borough : Literal['Manhattan','Queens','Bronx','Brooklyn']
    dropoff_borough : Literal['Manhattan','Queens','Brooklyn','Bronx','Staten Island']

class GenerateRespones(BaseModel):
    pred_total : float
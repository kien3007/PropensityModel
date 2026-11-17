"""
Feast Entity definitions for Household Purchase Propensity Platform.
"""
from feast import Entity
from feast.value_type import ValueType

household = Entity(
    name="household_key",
    join_keys=["household_key"],
    value_type=ValueType.INT64,
    description="Unique Dunnhumby household identifier",
)

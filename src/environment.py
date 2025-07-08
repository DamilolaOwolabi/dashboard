import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import numpy.typing as npt


class GridConstants:
    MAX_STORAGE_CAPACITY = 1000
    STORAGE_EFFICIENCY = 0.95
    STORAGE_RATE = 50  # MWh per timestep
    DEFAULT_STORAGE_LEVEL = 500
    BASE_ENERGY_PRICE = 50
    WIND_EFFICIENCY_RANGE = (0.8, 1.0)
    LOAD_RESPONSE_RANGE = (0.9, 1.1)
    TREND_WINDOW = 3
    MAX_PRICE_MULTIPLIER = 10.0
    MIN_PRICE_MULTIPLIER = 0.1


class WindGridEnv:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.wind_capacity = df['wind_generation'].values
        self.load_demand = df['load_demand'].values
        self.max_steps = len(df)
        self.agent_ids = ["wind_agent", "storage_agent", "load_agent"]
        self.action_size = 3
        self.storage_level = GridConstants.DEFAULT_STORAGE_LEVEL
        self.current_step = 0
        self._last_imbalance = 0.0

    def reset(self) -> Dict[str, npt.NDArray[np.float32]]:
        self.storage_level = GridConstants.DEFAULT_STORAGE_LEVEL
        self.current_step = 0
        self._last_imbalance = 0.0
        return self._get_observations()

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, npt.NDArray[np.float32]], Dict[str, float], bool, Dict[str, Any]
    ]:
        self._validate_actions(actions)

        wind_action = actions["wind_agent"]
        storage_action = actions["storage_agent"]
        load_action = actions["load_agent"]

        wind_eff = np.linspace(*GridConstants.WIND_EFFICIENCY_RANGE, 3)[wind_action]
        load_factor = [1.1, 1.0, 0.9][load_action]

        wind_output = self.wind_capacity[self.current_step] * wind_eff
        adjusted_load = self.load_demand[self.current_step] * load_factor
        storage_change = self._apply_storage_action(storage_action)

        storage_discharge = -storage_change if storage_change < 0 else 0
        net_supply = wind_output + storage_discharge
        imbalance = net_supply - adjusted_load
        self._last_imbalance = imbalance

        rewards = self._calculate_rewards(imbalance, wind_output, storage_change, adjusted_load)

        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_observations(), rewards, done, {}

    def _apply_storage_action(self, action: int) -> float:
        if action == 0:  # Discharge
            discharge = min(GridConstants.STORAGE_RATE, self.storage_level)
            self.storage_level -= discharge
            return discharge * GridConstants.STORAGE_EFFICIENCY
        elif action == 2:  # Charge
            available_capacity = GridConstants.MAX_STORAGE_CAPACITY - self.storage_level
            charge = min(GridConstants.STORAGE_RATE, available_capacity)
            energy_cost = charge / GridConstants.STORAGE_EFFICIENCY
            self.storage_level += charge
            return -energy_cost
        return 0.0  # Hold

    def _calculate_rewards(
        self, imbalance: float, wind_output: float, storage_change: float, adjusted_load: float
    ) -> Dict[str, float]:
        base_reward = -abs(imbalance)

        wind_capacity_now = self.wind_capacity[self.current_step] if self.current_step < self.max_steps else self.wind_capacity[-1]
        wind_utilization = wind_output / (wind_capacity_now + 1e-6)
        wind_reward = (wind_utilization - 0.9) * 0.1

        if self.storage_level < 0.1 * GridConstants.MAX_STORAGE_CAPACITY or self.storage_level > 0.9 * GridConstants.MAX_STORAGE_CAPACITY:
            storage_reward = -0.1
        else:
            storage_reward = 0.0

        base_load = self.load_demand[self.current_step] if self.current_step < self.max_steps else self.load_demand[-1]
        demand_response = abs(adjusted_load - base_load) / (base_load + 1e-6)
        load_reward = (
            demand_response * 0.05 if abs(imbalance) > 10 else -demand_response * 0.02
        )

        return {
            "wind_agent": base_reward + wind_reward,
            "storage_agent": base_reward + storage_reward,
            "load_agent": base_reward + load_reward,
        }

    def _get_observations(self) -> Dict[str, npt.NDArray[np.float32]]:
        wind = self.wind_capacity[self.current_step] if self.current_step < self.max_steps else 0.0
        load = self.load_demand[self.current_step] if self.current_step < self.max_steps else 0.0
        price = self._get_dynamic_price(wind, load)
        storage_ratio = self.storage_level / GridConstants.MAX_STORAGE_CAPACITY
        wind_trend = self._get_wind_trend()
        load_trend = self._get_load_trend()

        return {
            "wind_agent": np.array([wind, price, storage_ratio, wind_trend], dtype=np.float32),
            "storage_agent": np.array([storage_ratio, price, load, wind_trend], dtype=np.float32),
            "load_agent": np.array([load, price, wind, load_trend], dtype=np.float32),
        }

    def _get_dynamic_price(self, wind: float, load: float) -> float:
        demand_factor = max(0.1, load / (np.mean(self.load_demand) + 1e-6))
        supply_factor = max(0.1, wind / (np.mean(self.wind_capacity) + 1e-6))
        multiplier = np.clip(demand_factor / supply_factor, GridConstants.MIN_PRICE_MULTIPLIER, GridConstants.MAX_PRICE_MULTIPLIER)
        return GridConstants.BASE_ENERGY_PRICE * multiplier

    def _get_wind_trend(self) -> float:
        window = GridConstants.TREND_WINDOW
        if self.current_step < window:
            return 0.0
        recent = self.wind_capacity[self.current_step - window:self.current_step]
        if len(recent) < 2:
            return 0.0
        return float((recent[-1] - recent[0]) / (window - 1))

    def _get_load_trend(self) -> float:
        window = GridConstants.TREND_WINDOW
        if self.current_step < window:
            return 0.0
        recent = self.load_demand[self.current_step - window:self.current_step]
        if len(recent) < 2:
            return 0.0
        return float((recent[-1] - recent[0]) / (window - 1))

    def _validate_actions(self, actions: Dict[str, int]) -> None:
        for agent in self.agent_ids:
            if agent not in actions:
                raise ValueError(f"Missing action for {agent}. Required: {self.agent_ids}")
            if actions[agent] not in [0, 1, 2]:
                raise ValueError(f"Invalid action {actions[agent]} for {agent}. Must be 0, 1, or 2.")

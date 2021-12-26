import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.visualization.UserParam import UserSettableParameter                                               
import scipy.stats as ss


# Simulation model parameters
# -------------------------------------------------------------------------------------------
model_params = {
    'no_agents': UserSettableParameter(
        'number', 'Number of agents', 100, 5, 5000, 5),
    'width': 50,
    'height': 50,
    'init_infected': UserSettableParameter(
        'slider', '% of initial pop. infected', 0.2, 0, 1, 0.05),
    'perc_masked': UserSettableParameter(
        'slider', '% masked', 0.5, 0, 1, 0.05),
    'prob_trans_masked': UserSettableParameter(
        'slider', 'Transmission prob. masked', 0.25, 0, 1, 0.05),
    'prob_trans_unmasked': UserSettableParameter(
        'slider', 'Transmission prob. unmasked', 0.75, 0, 1, 0.05),
    'infection_period': UserSettableParameter(
        'slider', '# simulation steps to move from infection to recovery', 50, 5, 200, 5),
    'immunity_period': UserSettableParameter(
        'slider', '# simulation steps before immunity is gone', 200, 10, 1000, 10)
}


# Agent class in Mesa
# -------------------------------------------------------------------------------------------
class Agent(Agent):
    """Agents in the CovidModel class below"""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.masked = bool(ss.bernoulli.rvs(self.model.perc_masked))
        self.infected = bool(ss.bernoulli.rvs(self.model.init_infected))
        self.immune = False
        self.recovery_countdown = 0
        self.immunity_countdown = 0
        # Random recovery countdown considers agents got infected at different times
        if self.infected:
            self.recovery_countdown = np.random.randint(1, self.model.infection_period + 1)

    def move(self):
        x, y = self.pos
        new_x = min(max(np.random.choice(
            [-1, 0, 1]) + x, 0), self.model.grid.height - 1)
        new_y = min(max(np.random.choice(
            [-1, 0, 1]) + y, 0),  self.model.grid.height - 1)
        self.model.grid.move_agent(self, (new_x, new_y))

    def update_infected(self):
        # Infected or immune agents cannot become infected
        if self.infected | self.immune:
            return None
        # Workaround for potential bug
        pos = tuple([int(self.pos[0]), int(self.pos[1])])
        # List of agents in the same grid cell
        cell_agents = self.model.grid.get_cell_list_contents(pos)
        # Checks if any of the agents in the cell are infected
        any_infected = any(a.infected for a in cell_agents)
        if any_infected:
            if self.masked:
                # Probability of getting infected when masked
                self.infected = bool(ss.bernoulli.rvs(
                    self.model.prob_trans_masked))
            elif ~self.masked:
                # Probability of getting infected when not masked
                self.infected = bool(ss.bernoulli.rvs(
                    self.model.prob_trans_unmasked))
        # Once infected countdown to recovery begins
        if self.infected:
            self.recovery_countdown = self.model.infection_period

    def update_recovered(self):
        if self.recovery_countdown == 1:
            self.infected = False
            self.immune = True
            # After recovery countdown to immunity going away begins
            self.immunity_countdown = self.model.immunity_period
        if self.recovery_countdown > 0:
            self.recovery_countdown += -1

    def update_susceptible(self):
        # After immunity wanes away, agent becomes susceptible
        if self.immunity_countdown == 1:
            self.immune = False
        if self.immunity_countdown > 0:
            self.immunity_countdown = self.immunity_countdown - 1

    def step(self):
        self.move()
        self.update_infected()
        self.update_recovered()
        self.update_susceptible()


# Model class in Mesa
# -------------------------------------------------------------------------------------------
class CovidModel(Model):
    """A SIR-like model with a number of agents that potentially transmit COVID
    when they are on the same cell of the grid"""

    def __init__(self, no_agents, width, height,
                 init_infected, perc_masked, prob_trans_masked,
                 prob_trans_unmasked, infection_period, immunity_period):
        self.no_agents = no_agents
        self.grid = MultiGrid(width, height, True)
        self.init_infected = init_infected
        self.perc_masked = perc_masked
        self.prob_trans_masked = prob_trans_masked
        self.prob_trans_unmasked = prob_trans_unmasked
        self.infection_period = infection_period
        self.immunity_period = immunity_period
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.no_agents):
            a = Agent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Collect count of susceptible, infected, and recovered agents
        self.datacollector = DataCollector({
            'Susceptible': 'susceptible',
            'Infected': 'infected',
            'Recovered & Immune': 'immune'})

    @property
    def susceptible(self):
        agents = self.schedule.agents
        susceptible = [not(a.immune | a.infected) for a in agents]
        return int(np.sum(susceptible))

    @property
    def infected(self):
        agents = self.schedule.agents
        infected = [a.infected for a in agents]
        return int(np.sum(infected))

    @property
    def immune(self):
        agents = self.schedule.agents
        immune = [a.immune for a in agents]
        return int(np.sum(immune))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

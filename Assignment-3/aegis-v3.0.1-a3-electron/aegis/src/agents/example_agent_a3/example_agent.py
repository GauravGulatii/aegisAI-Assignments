"""
example_agent.py

Submitted by: 
Group 24
Gaurav Gulati - 30121866
Miguel Fuentes - 
Sukhnaaz Sidhu - 
Zahra Ali - 

Course: CPSC 383 (Fall 2025)

References:
- This assignment partially reuses code from my A1 solution for pathfinding.
- (https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- Basic partial logic from assignment 1 was integrated here for A3.

Part-1 successfully implemented
"""

import heapq
from typing import override

from aegis import (
    END_TURN,
    SEND_MESSAGE_RESULT,
    MOVE,
    OBSERVE_RESULT,
    PREDICT_RESULT,
    SAVE_SURV,
    SAVE_SURV_RESULT,
    SEND_MESSAGE,
    TEAM_DIG,
    AgentCommand,
    AgentIDList,
    Direction,
    Rubble,
    Survivor,
    Location,
    World,
)
from a3.agent import BaseAgent, Brain, AgentController


#Chebyshev distance = max(|a.x - b.x|, |a.y - b.y|)
def chebyshev_distance(a: Location, b: Location) -> float:
    diff_x = abs(a.x - b.x)
    diff_y = abs(a.y - b.y)
    return max(diff_x, diff_y)

# Small wrapper around Python heapq for storing (priority, item)
class SimplePriorityQueue:
    def __init__(self) -> None:
        self._heap = []

    def put(self, item, priority):
        heapq.heappush(self._heap, (priority, item))

    def get(self):
        # Return the item with smallest priority
        prior, itm = heapq.heappop(self._heap)
        return itm

    def empty(self) -> bool:
        return len(self._heap) == 0


class ExampleAgent(Brain):
    def __init__(self) -> None:
        super().__init__()
        self._agent: AgentController = BaseAgent.get_agent()
        
        # Track of places we have visited
        self._visited: set[Location] = set()
        # Also track known move costs, from the local observations
        self._move_costs = {}

    @override
    def handle_send_message_result(self, smr: SEND_MESSAGE_RESULT) -> None:
        self._agent.log(f"SEND_MESSAGE_RESULT: {smr}")
        self._agent.log(f"{smr}")
        #print("#--- You need to implement handle_send_message_result function! ---#")

    @override
    def handle_observe_result(self, ovr: OBSERVE_RESULT) -> None:
        self._agent.log(f"OBSERVE_RESULT: {ovr}")
        self._agent.log(f"{ovr}")
        #print("#--- You need to implement handle_observe_result function! ---#")
        

    @override
    def handle_save_surv_result(self, ssr: SAVE_SURV_RESULT) -> None:
        self._agent.log(f"SAVE_SURV_RESULT: {ssr}")
        self._agent.log(f"{ssr}")
        #print("#--- You need to implement handle_save_surv_result function! ---#")

    @override
    def handle_predict_result(self, prd: PREDICT_RESULT) -> None:
        self._agent.log(f"PREDICT_RESULT: {prd}")
        self._agent.log(f"{prd}")

    @override
    def think(self) -> None:
        self._agent.log("Thinking...")

        # Send a message to other agents in my group.
        # Empty AgentIDList will send to group members.
        self._agent.send(
            SEND_MESSAGE(
                AgentIDList(), f"Hello from agent {self._agent.get_agent_id().id}"
            )
        )
        
        # Retrieve the current state of the world.
        world = self.get_world()
        if world is None:
            self.send_and_end_turn(MOVE(Direction.CENTER))
            return

        my_loc = self._agent.get_location()
        cell_here = world.get_cell_at(my_loc)
        if cell_here is None:
            self.send_and_end_turn(MOVE(Direction.CENTER))
            return

        # Update local knowledge about move costs around
        self.update_local_costs(world, my_loc)

         # Get the top layer at the agent’s current location.
        top_layer = cell_here.get_top_layer()

        # If a survivor is present, save it and end the turn.
        if isinstance(top_layer, Survivor):
            self.send_and_end_turn(SAVE_SURV())
            return

        # If rubble is present, clear it and end the turn.
        if isinstance(top_layer, Rubble):
            self.send_and_end_turn(TEAM_DIG())
            return

        # otherwise see if we know about a location that has a survivor
        known_surv_loc = self.find_known_survivor(world)
        if known_surv_loc is not None:
            path = self.a_star_path(world, my_loc, known_surv_loc)
            if len(path) > 1:
                # path[0] is current, path[1] is next step
                direction = my_loc.direction_to(path[1])
                self.send_and_end_turn(MOVE(direction))
                return
            else:
                # no real path
                self.send_and_end_turn(MOVE(Direction.CENTER))
                return

        # if no known survivors, we can try exploring new unvisited places
        target_spot = self.pick_unvisited_cell(world, my_loc)
        if target_spot is not None:
            path = self.a_star_path(world, my_loc, target_spot)
            if len(path) > 1:
                direction = my_loc.direction_to(path[1])
                self.send_and_end_turn(MOVE(direction))
                return

        # fallback if no target
        self.send_and_end_turn(MOVE(Direction.CENTER))

    # Helper methods

    def pick_unvisited_cell(self, world: World, start: Location):
        frontier = SimplePriorityQueue()
        frontier.put(start, 0)
        cost_so_far = {start: 0}
        
        directions = list(Direction)

        while not frontier.empty():
            cur = frontier.get()
            
            # check if its unvisited
            if cur not in self._visited:
                return cur  # found an unvisited location

            for d in directions:
                nxt = cur.add(d)
                if not world.on_map(nxt):
                    continue
                c = world.get_cell_at(nxt)
                if c is None or c.is_fire_cell() or c.is_killer_cell():
                    continue

                # cost for neighbor
                mc = self._move_costs.get(nxt, c.move_cost)
                new_c = cost_so_far[cur] + mc
                if (nxt not in cost_so_far) or (new_c < cost_so_far[nxt]):
                    cost_so_far[nxt] = new_c
                    frontier.put(nxt, new_c)

        # no unvisited cell found
        return None

# return any location in world grid that has_survivors == True
    def find_known_survivor(self, world: World):
        for row in world.get_world_grid():
            for c in row:
                if c.has_survivors:
                    return c.location
        return None

# for the center cell plus neighbors, store their move_cost in self._move_costs.
    def update_local_costs(self, world: World, center: Location):
        directions = list(Direction)
        for d in directions:
            loc = center.add(d)
            if world.on_map(loc):
                c = world.get_cell_at(loc)
                if c is not None:
                    self._move_costs[loc] = c.move_cost
        # also mark center visited
        self._visited.add(center)

# A star algorithm referenced from A1 Submission and https://www.redblobgames.com/pathfinding/a-star/introduction.html
    def a_star_path(self, world: World, start: Location, goal: Location) -> list[Location]:
        frontier = SimplePriorityQueue()
        frontier.put(start, 0)
        came_from = {start: None}
        cost_so_far = {start: 0}

        directions = list(Direction)

        while not frontier.empty():
            current = frontier.get()
            if current == goal:
                break

            for d in directions:
                nxt = current.add(d)
                if not world.on_map(nxt):
                    continue
                c = world.get_cell_at(nxt)
                if c is None or c.is_fire_cell() or c.is_killer_cell():
                    continue

                # cost to step
                mc = self._move_costs.get(nxt, c.move_cost)
                new_cost = cost_so_far[current] + mc
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + chebyshev_distance(nxt, goal)
                    frontier.put(nxt, priority)
                    came_from[nxt] = current

        if goal not in came_from:
            return [start]  # no route

        return self.reconstruct_path(came_from, start, goal)

    def reconstruct_path(self, came_from: dict, start: Location, goal: Location) -> list[Location]:
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    def send_and_end_turn(self, command: AgentCommand):
        """Send a command and end your turn."""
        self._agent.log(f"SENDING {command}")
        self._agent.send(command)
        self._agent.send(END_TURN())

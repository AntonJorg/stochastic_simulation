import numpy as np
import pandas as pd
from enum import IntEnum
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from scipy.stats import expon


class Buffer(IntEnum):
    DIRTY = 0
    CLEAN = 1


def decorate_event(cls):
    return dataclass(slots=True, frozen=True)(cls)


@decorate_event
class Event:
    time: float

    def __lt__(self, other):
        return self.time < other.time
    
    # __repr__ didnt work
    def print(self):
        total_seconds = int(self.time * 3600)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"[{hours:02}:{minutes:02}:{seconds:02}] {self}")


@decorate_event
class PatientArrived(Event):
    pass


@decorate_event
class PatientDischarged(Event):
    pass


@decorate_event
class BedArrivedElevator(Event):
    elevator_id: int
    buffer: Buffer

@decorate_event
class BedArrivedWashing(Event):
    pass

@decorate_event
class BedStartedWashing(Event):
    pass

@decorate_event
class BedFinishedWashing(Event):
    pass

@decorate_event
class RobotsUpdated(Event):
    pass

@decorate_event
class RobotsDispatched(Event):
    robot_id: int

@decorate_event
class PickUpBed(Event):
    robot_id: int
    location_id: int
    buffer: Buffer

@decorate_event
class DropOffBed(Event):
    robot_id: int
    location_id: int
    buffer: Buffer


@dataclass
class SimulationParams:
    discharge_dist : callable
    service_time_dist : callable
    transport_time_dist : callable
    distance_matrix: np.ndarray
    arrival_weights: np.ndarray
    discharge_weights: np.ndarray
    n_elevators: int
    n_robots: int


def simulate_system(params: SimulationParams, verbose=False):

    discharge_dist = params.discharge_dist
    service_time_dist = params.service_time_dist
    transport_time_dist = params.transport_time_dist
    distance_matrix = params.distance_matrix
    n_elevators = params.n_elevators
    n_robots = params.n_robots
    arrival_weights = params.arrival_weights
    discharge_weights = params.discharge_weights

    # 0 = washer
    # 1 = elevator 1
    # 2 = elevator 2
    # ...

    buffers = np.zeros((2, 1 + n_elevators))
    reserved = np.zeros_like(buffers)
    demand = np.zeros(1 + n_elevators) # index 0 never used, but keeps indexing consistent

    robot_location = np.zeros(n_robots, dtype=int)
    robot_traveling = np.zeros(n_robots, dtype=bool)
    robot_beds = np.zeros(n_robots, dtype=bool)

    discharge_times = discharge_dist()
    arrival_times = discharge_times + expon(scale=1).rvs(size=discharge_times.shape[0])

    events = [PatientDischarged(time) for time in discharge_times]
    events += [PatientArrived(time) for time in arrival_times]
    heapify(events)

    events_processed = []
    buffer_list = [buffers.copy()]
    demand_list = [demand.copy()]
    times = [0.0]

    washer_ready = True

    while events:
    
        event = heappop(events)
        
        if verbose:
            event.print()

        match event:
            case PatientArrived(time=t):
                eid = np.random.choice(n_elevators, p=arrival_weights) + 1
                if buffers[Buffer.CLEAN, eid] > 0:
                    buffers[Buffer.CLEAN, eid] -= 1
                else:
                    demand[eid] += 1

            case PatientDischarged(time=t):
                eid = np.random.choice(n_elevators, p=discharge_weights) + 1
                heappush(events, BedArrivedElevator(t, eid, Buffer.DIRTY))
            
            case BedArrivedElevator(time=t, elevator_id=eid, buffer=buffer):
                buffers[buffer, eid] += 1
                heappush(events, RobotsUpdated(t))
        
            case BedArrivedWashing(time=t):
                # If washer ready, start washing immediately 
                if washer_ready:
                    heappush(events, BedStartedWashing(t))
                
            case BedStartedWashing(time=t):
                buffers[Buffer.DIRTY, 0] -= 1
                washer_ready = False
                heappush(events, BedFinishedWashing(t + service_time_dist()))

            case BedFinishedWashing(time=t):
                buffers[Buffer.CLEAN, 0] += 1

                heappush(events, RobotsUpdated(t))

                washer_ready = True

                # If washer is ready and there are beds waiting outside
                if buffers[Buffer.DIRTY, 0] > 0:
                    heappush(events, BedStartedWashing(t))
                
            
            case RobotsUpdated(time=t):
                available_robots = np.where(~robot_traveling)[0]
                for robot_id in available_robots:
                    dispatched = False
                    available_beds = buffers - reserved
                    # handle dirty beds first
                    if np.any(available_beds[Buffer.DIRTY, 1:] != 0):
                        location_id = np.argmax(available_beds[Buffer.DIRTY, 1:]) + 1
                        robot_traveling[robot_id] = True

                        reserved[Buffer.DIRTY, location_id] += 1

                        transport_time = transport_time_dist(distance_matrix[robot_location[robot_id], location_id])
                        heappush(events, RobotsDispatched(t, robot_id))
                        heappush(events, PickUpBed(t + transport_time, robot_id, location_id, Buffer.DIRTY))    
                        dispatched = True

                    if not dispatched and available_beds[Buffer.CLEAN, 0] != 0:
                        location_id = 0
                        robot_traveling[robot_id] = True

                        reserved[Buffer.CLEAN, location_id] += 1

                        transport_time = transport_time_dist(distance_matrix[robot_location[robot_id], location_id])
                        heappush(events, RobotsDispatched(t, robot_id))
                        heappush(events, PickUpBed(t + transport_time, robot_id, location_id, Buffer.CLEAN))
                    #print(robot_location)
                    #print(buffers)

            case PickUpBed(time=t, robot_id=robid, location_id=locid, buffer=buffer):
                assert buffers[buffer, locid] > 0, "Must have beds to pick up!"
                buffers[buffer, locid] -= 1
                reserved[buffer, locid] -= 1
                robot_location[robid] = locid
                robot_beds[robid] = True

                if buffer == Buffer.DIRTY:
                    target_location = 0
                else:
                    expected_time_to_depletion = buffers[Buffer.CLEAN, 1:] / arrival_weights
                    min_time = np.min(expected_time_to_depletion)

                    indices = np.where(expected_time_to_depletion == min_time)[0]
                    target_location = np.random.choice(indices) + 1

                transport_time = transport_time_dist(distance_matrix[locid, target_location])

                heappush(events, DropOffBed(t + transport_time, robid, target_location, buffer))

            case DropOffBed(time=t, robot_id=robid, location_id=locid, buffer=buffer):
                assert robot_beds[robid], "Robots must have bed to drop off"
                
                buffers[buffer, locid] += 1
                robot_location[robid] = locid
                robot_beds[robid] = False
                robot_traveling[robid] = False

                if locid == 0: # dropped off at washer
                    heappush(events, BedArrivedWashing(t))
                else: # dropped off at elevator
                    if demand[locid] > 0:
                        buffers[buffer, locid] -= 1
                        demand[locid] -= 1

                heappush(events, RobotsUpdated(t))

            case RobotsDispatched():
                # only for animation purposes
                pass

            case _:
                raise ValueError("Unknown event type!")

        buffer_list.append(buffers.copy())
        demand_list.append(demand.copy())
        times.append(event.time)

        if verbose:
            print("Buffers")
            print(buffers)
            print("Demand")
            print(demand)
            print("Robots")
            print(robot_location)
            print(robot_traveling)   
            print()         

        events_processed.append(event)

    assert np.all(buffers[Buffer.DIRTY] == 0)

    data = {
        "buffers": np.stack(buffer_list, axis=-1),
        "demands": np.stack(demand_list, axis=-1),
        "times": np.array(times),
        "arrivals": arrival_times,
        "discharges": discharge_times,
    }

    return events_processed, data

def simulate_many(params: SimulationParams, n_iters: int):
    n_points = 24 * 4 + 1

    max_clean_buffer_sizes = np.empty(n_iters)
    max_dirty_buffer_sizes = np.empty(n_iters)
    max_demand = np.empty(n_iters)

    buffers = np.empty((n_iters, 2, params.n_elevators + 1, n_points))
    demands = np.empty((n_iters, params.n_elevators + 1, n_points))

    arrival_times = []
    discharge_times = []

    for n in range(n_iters):
        _, data = simulate_system(params)

        times = data["times"]
        # New time range
        new_time = np.linspace(0, 24, n_points)

        # Find indices for forward fill
        indices = np.searchsorted(times, new_time, side='right') - 1

        buffers[n] = data["buffers"][:, :, indices] 
        demands[n] = data["demands"][:, indices]

        arrival_times.append(data["arrivals"])
        discharge_times.append(data["discharges"])

        max_clean_buffer_sizes[n] = np.max(data["buffers"][Buffer.CLEAN])
        max_dirty_buffer_sizes[n] = np.max(data["buffers"][Buffer.DIRTY])
        max_demand[n] = np.max(data["demands"])

    data = {
        "Clean buffer (daily max)": max_clean_buffer_sizes,
        "Dirty buffer (daily max)": max_dirty_buffer_sizes,
        "Bed demand (daily max)": max_demand,
    }

    df = pd.DataFrame(
        {name: np.percentile(arr, [50, 90, 95, 99]) for name, arr in data.items()},
        index=['50th', '90th', '95th', '99th']
    ).T

    return buffers, demands, np.concatenate(arrival_times), np.concatenate(discharge_times)

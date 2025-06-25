import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from heapq import heapify, heappop, heappush


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
    arrival_dist: callable
    service_time_dist : callable
    transport_time_dist : callable
    distance_matrix: np.ndarray
    arrival_weights: np.ndarray
    discharge_weights: np.ndarray
    n_patients: int
    n_elevators: int
    n_robots: int


def simulate_system(params: SimulationParams):

    discharge_dist = params.discharge_dist
    arrival_dist = params.arrival_dist
    service_time_dist = params.service_time_dist
    transport_time_dist = params.transport_time_dist
    distance_matrix = params.distance_matrix
    n_patients = params.n_patients
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

    discharge_times = discharge_dist.rvs(size=n_patients)
    arrival_times = arrival_dist.rvs(size=n_patients)

    events = [PatientDischarged(time) for time in discharge_times]
    events += [PatientArrived(time) for time in arrival_times]
    heapify(events)

    events_processed = []
    buffer_list = [buffers.copy()]
    demand_list = [demand.copy()]
    times = [0.0]

    washer_ready = True

    iterations = 0
    while events:
        iterations += 1
        if iterations > 10000:
            break

        event = heappop(events)
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
                        heappush(events, PickUpBed(t + transport_time, robot_id, location_id, Buffer.DIRTY))    
                        dispatched = True

                    if not dispatched and available_beds[Buffer.CLEAN, 0] != 0:
                        location_id = 0
                        robot_traveling[robot_id] = True

                        reserved[Buffer.CLEAN, location_id] += 1

                        transport_time = transport_time_dist(distance_matrix[robot_location[robot_id], location_id])
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

            case _:
                raise ValueError("Unknown event type!")

        buffer_list.append(buffers.copy())
        demand_list.append(demand.copy())
        times.append(event.time)

        verbose = True
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

    data = {
        "buffers": np.stack(buffer_list, axis=-1),
        "demands": np.stack(demand_list, axis=-1),
        "times": np.array(times),
        "arrivals": arrival_times,
        "discharges": discharge_times,
    }

    return events_processed, data
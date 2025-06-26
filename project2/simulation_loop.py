import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def simulate_system(params: SimulationParams, verbose=False, lambda_arrival=1.0):

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
    arrival_times = discharge_times + expon(scale=lambda_arrival).rvs(size=discharge_times.shape[0])

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
                    robot_loc = robot_location[robot_id]
                    dispatched = False
                    available_beds = buffers - reserved

                    # Pick up bed when traveling to 
                    if available_beds[:, robot_loc].sum() > 0:
                        target_buffer = Buffer.CLEAN if robot_loc == 0 else Buffer.DIRTY
                        if available_beds[target_buffer, robot_loc] > 0:
                            transport_time = transport_time_dist(distance_matrix[robot_loc, robot_loc])
                            reserved[target_buffer, robot_loc] += 1
                            heappush(events, RobotsDispatched(t, robot_id))
                            heappush(events, PickUpBed(t + transport_time, robot_id, robot_loc, target_buffer))    
                            robot_traveling[robot_id] = True
                            dispatched = True

                    # handle dirty beds first
                    if not dispatched and np.any(available_beds[Buffer.DIRTY, 1:] != 0):
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
from tqdm import tqdm
def simulate_many(params: SimulationParams, n_iters: int, **kwargs):
    n_points = 24 * 12 + 1

    max_clean_buffer_sizes = np.empty(n_iters)
    max_dirty_buffer_sizes = np.empty(n_iters)
    max_demand = np.empty(n_iters)

    buffers = np.empty((n_iters, 2, params.n_elevators + 1, n_points))
    demands = np.empty((n_iters, params.n_elevators + 1, n_points))

    arrival_times = []
    discharge_times = []

    for n in tqdm(range(n_iters)):
        _, data = simulate_system(params, lambda_arrival=kwargs.get("lambda_arrival", 1.0))

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

    return buffers, demands, np.concatenate(arrival_times), np.concatenate(discharge_times), df


import numpy as np
import matplotlib.pyplot as plt

def plot_result(params, buffers, demands, arrivals, discharges, filename, title):
    plt.rcParams.update({'font.size': 14}) 
    fig, ax = plt.subplots(3, params.n_elevators + 1, figsize=(20, 8), sharex=True)

    t = np.linspace(0, 24, buffers.shape[-1])
    bufnames = ["Dirty", "Clean"]

    max_percentiles = np.empty((2, params.n_elevators + 1))

    for buftype in [0, 1]:
        for i in range(params.n_elevators + 1):
            placename = "Washer" if i == 0 else f"Elevator {i}"
            buf = buffers[:, buftype, i, :]

            ax[buftype, i].set_title(f"{placename}, {bufnames[buftype]} Buffer")
            # Compute 2.5th and 97.5th percentiles for 95% interval
            lower = np.percentile(buf, 2.5, axis=0)
            upper = np.percentile(buf, 97.5, axis=0)
            upper_99 = np.percentile(np.max(buf, axis=1), 99)
            max_percentiles[buftype, i] = upper_99
            mean = np.mean(buf, axis=0)
            #sem = np.std(buf, axis=0, ddof=1) / np.sqrt(buf.shape[0])
            #ci95 = 1.96 * sem
            #ax[buftype, i].fill_between(t, mean - ci95, mean + ci95, alpha=0.5)
            ax[buftype, i].axhline(upper_99, color='red', linestyle='--', label='99th percentile of maximum')
            ax[buftype, i].plot(t, mean, label="Mean")
            ax[buftype, i].fill_between(t, lower, upper, alpha=0.5, label='95% interval')
            ax[buftype, i].grid(True, axis='y', linestyle='--', alpha=0.5)


    column_labels = ["Bed Wash"] + [f"Elevator {i+1}" for i in range(params.n_elevators)]
    df = pd.DataFrame(max_percentiles, index=bufnames, columns=column_labels)
    print(df)

    with open(f"results/{filename}.tex", "w") as f:
        latex = df.to_latex(index=True)
        f.write(latex)

    ax[0, params.n_elevators].legend()

    for i in range(1, params.n_elevators + 1):
        buf = demands[:, i, :]
        lower = np.percentile(buf, 2.5, axis=0)
        upper = np.percentile(buf, 97.5, axis=0)
        mean = np.mean(buf, axis=0)
        #sem = np.std(buf, axis=0, ddof=1) / np.sqrt(buf.shape[0])
        #ci95 = 1.96 * sem
        ax[2, i].set_title(f"Elevator {i}, Bed Demand")
        ax[2, i].plot(t, mean, label="Mean")
        ax[2, i].fill_between(t, lower, upper, alpha=0.5, label='95% interval')
        #ax[2, i].fill_between(t, mean - ci95, mean + ci95, alpha=0.5)
        ax[2, i].grid(True, axis='y', linestyle='--', alpha=0.5)


    ax[2, params.n_elevators].legend()

    ax[2, 0].hist(arrivals, bins=24, alpha=0.5, label="Arrivals")
    ax[2, 0].hist(discharges, bins=24, alpha=0.5, label="Discharges")
    ax[2, 0].legend()
    ax[2, 0].set_title("Arrivals and Discharges")
    ax[2, 0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Add a common x-axis label
    fig.text(0.5, 0.04, "Time (hours)", ha='center')

    # Optional super title
    fig.suptitle(f"{title}: System Buffers and Patient Flow Over Time", fontsize=18)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/{filename}.pdf", format="pdf")
    plt.show()



if __name__ == "__main__":
    from scipy.stats import norm, gamma, uniform
    import pickle

    def service_time_dist():
        service_min_time = 5/60
        return service_min_time + expon(scale=1/60).rvs()


    def transport_time_dist(distance_meters):
        if distance_meters == 0:
            return 0
        speed_meters_per_second = 1.4
        transport_time_seconds = gamma(a=distance_meters, scale=1/speed_meters_per_second).rvs()
        return transport_time_seconds / 3600

    distances = np.array([
        [10, 175, 0],
        [175, 10, 175],
        [0, 175, 10],
    ])

    discharge_dist = lambda: np.cumsum(expon(scale=1/12).rvs(size=270))
    n_elevators = 1

    params = SimulationParams(
        discharge_dist,
        service_time_dist,
        transport_time_dist,
        distances,
        arrival_weights=np.ones(n_elevators) / n_elevators,
        discharge_weights=np.ones(n_elevators) / n_elevators,
        n_elevators=n_elevators,
        n_robots=1,
    )

    # events, data = simulate_system(params)

    buffers, demands, arrival_times, discharge_times, df = simulate_many(params, n_iters=10)
    plot_result(params, buffers, demands, arrival_times, discharge_times)
    
    # with open("events.pkl", "wb") as f:
    #     pickle.dump(events, f)
    # print(events)
    # cd C:/Users/Jason/OneDrive - Danmarks Tekniske Universitet/Masters/2_Semester/Stochastic_Simulation/stochastic_simulation/project2/
    # python simulation_loop.py

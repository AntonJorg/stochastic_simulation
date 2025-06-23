from dataclasses import dataclass
from heapq import heappush, heappop, heapify
import numpy as np
from scipy.stats import expon, norm, uniform
import matplotlib.pyplot as plt

p = 0.55
t = np.linspace(0, 24, 1000)

class ArrivalDistribution:
    first_round_time = 9
    second_round_time = 15
    def __init__(self, p=0.5, offset=0):
        self.p = p
        self.first_peak = norm(loc=10 + offset, scale=1)
        self.second_peak = norm(loc=15 + offset, scale=1.5)

    def pdf(self, x):
        return self.p * self.first_peak.pdf(x) + (1 - self.p) * self.second_peak.pdf(x)

    def rvs(self, size=()):
        s1 = self.first_peak.rvs(size=size)
        s2 = self.second_peak.rvs(size=size)
        choice = uniform.rvs(size=size) < self.p
        return np.where(choice, s1, s2)

discharge_dist = ArrivalDistribution()
admission_dist = ArrivalDistribution(offset=1)

@dataclass
class Event:
    time: float

    def __lt__(self, other):
        return self.time < other.time

@dataclass
class PatientAdmitted(Event):
    ward_id: int

@dataclass
class PatientDischarged(Event):
    ward_id: int

@dataclass
class BedLeaveElevator(Event):
    pass

@dataclass
class BedArrivedWashing(Event):
    pass

@dataclass
class BedFinishedWashing(Event):
    pass

@dataclass
class BedArriveElevator(Event):
    pass


class ButtManager:
    def __init__(self, num_bots: int, transport_time_dist : callable):
        self.bots_available = num_bots
        self.transport_time_dist = transport_time_dist
    
    def update(self, t, events, state):
        event_type = self.decision_policy(state) 
        if event_type is not None:
            event = event_type(t + self.transport_time_dist())
            heappush(events, event)
    
    def decision_policy(self, state):
        # Favor returning clean bed over starting washing
        if self.bots_available == 0:
            return None
        
        self.bots_available -= 1
        if state[2] > 0:
            state[2] -= 1
            return BedArriveElevator
        if state[0] > 0:
            state[0] -= 1
            return BedArrivedWashing
        
        raise ValueError("There should be at least one bed in either dirty or clean buffer to process.")

    def release_bot(self):
        self.bots_available += 1


def simulate_system(
        admission_dist : callable,
        discharge_dist : callable,
        service_time_dist : callable,
        transport_time_dist : callable,
        n_patients: int = 200
    ):

    state = [0] * 3  # [waiting for bot -> washing, waiting for washing, waiting for bot -> elevator]
    butt_manager = ButtManager(1, transport_time_dist)

    admission_times = admission_dist.rvs(size=n_patients)
    discharge_times = discharge_dist.rvs(size=n_patients)

    # events = [PatientAdmitted(time, 0) for time in admission_times]
    events = [BedLeaveElevator(time) for time in discharge_times]
    heapify(events)

    events_processed = []
    states = []
    times = []

    washer_ready = True

    iterations = 0
    while events:
        iterations += 1
        if iterations > 10000:
            break
        event = heappop(events)

        match event:
            case BedLeaveElevator(time=t):
                state[0] += 1
                butt_manager.update(t, events, state)

            case BedArrivedWashing(time=t):
                state[1] += 1
                butt_manager.release_bot()
                
                # If washer ready, start washing immediately 
                if washer_ready:
                    washer_ready = False
                    state[1] -= 1
                    heappush(events, BedFinishedWashing(t + service_time_dist()))
                
                butt_manager.update(t, events, state)
                    
            case BedFinishedWashing(time=t):
                washer_ready = True
                state[2] += 1
                butt_manager.release_bot()

                # If washer is ready and there are beds waiting outside
                if state[1] > 0:
                    washer_ready = False
                    state[1] -= 1
                    heappush(events, BedFinishedWashing(t + service_time_dist()))

                butt_manager.update(t, events, state)

            case BedArriveElevator(time=t):
                butt_manager.release_bot()
        
        states.append(state)
        times.append(event.time)
        events_processed.append(event)

    return events_processed, np.array(times), np.array(states, dtype=int)

# np.random.seed(42)
service_min_time = 1/120  # 10 minute
service_time_dist = lambda: service_min_time + expon(scale=1e-10).rvs()
transport_time_dist = lambda: norm(loc=1/120, scale=1e-2).rvs()  # 1/2 minute to transport a bed

results = simulate_system(admission_dist, discharge_dist, service_time_dist, transport_time_dist, n_patients=200)
print(f"{np.max(results[2])}")  # 10

def plot_queue_length(results):
    plt.figure(figsize=(12, 6))
    plt.title('Queue Length Over Time in Blocking System')
    # for run in results:
    plt.plot(results[1], results[2], label='Queue Length')
    
    plt.xlabel('Time')
    plt.ylabel('Queue Length')
    plt.show()

plot_queue_length(results)

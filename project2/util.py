from enum import IntEnum
from dataclasses import dataclass

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


########################### 
######## Animations #######
###########################

@decorate_event
class Anim:
    start : float
    duration : float

@decorate_event
class ElevatorAnim(Anim):
    elevator_id: int

@decorate_event
class BotAnim(Anim):
    bot_id: int
    location_id: int
    buffer: Buffer

@decorate_event
class WasherAnim(Anim):
    pass
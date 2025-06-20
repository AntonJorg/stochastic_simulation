from dataclasses import dataclass

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

import pygame as pg
from pygame.math import Vector2
import pickle
from util import *

BLACK = (24, 24, 24)
WHITE = (200, 200, 200)
RED = (255, 70, 70)
GREEN = (24, 200, 24)
BLUE = (24, 100, 255)

WIDTH, HEIGHT = 800, 600
FPS = 30
DT = 1 / FPS


class Bot:
    def __init__(self, id, position):
        self.id = id
        self.pos = position
        self.start_pos = self.pos.copy()
        self.bed = 0  # {0 : no bed, 1 : dirty bed, 2 : clean bed}
        self.running = False

    def start_animation(self, duration: float, goal_pos: Vector2, bed: int):
        self.elapsed = 0.0
        self.state = 0.0
        self.duration = duration
        self.goal_pos = goal_pos
        self.bed = bed
        self.start_pos = self.pos.copy()
        self.running = True

    def step(self):
        """Return whether the bot has reached its goal."""
        if not self.running:
            return True
        self.elapsed += DT
        if self.elapsed < self.duration:
            state = self.elapsed / self.duration
            self.pos = self.start_pos.lerp(self.goal_pos, state)
            return False

        self.pos = self.goal_pos
        self.running = False
        return True

class Animation:
    def __init__(self):
        self.state = 0.0
        self.running = False

    def start_animation(self, duration: float):
        self.elapsed = 0.0
        self.duration = duration
        self.running = True

    def step(self):
        """Return whether the animation is done."""
        if not self.running:
            return True
        
        self.elapsed += DT
        if self.elapsed < self.duration:
            self.state = self.elapsed / self.duration
            return False

        self.state = 0.0
        self.running = False
        return True

class HospitalSimulation:

    def __init__(self, events):
        self.animations = self._compile(events)
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Hospital Simulation")
        self.clock = pg.time.Clock()

        self.positions = [
            Vector2(WIDTH // 2, HEIGHT // 2 - 40),  # Washing machine
            Vector2(WIDTH // 2, HEIGHT // 8 + 65),  # Elevator top
        ]

        self.bots = [Bot(i, self.positions[0].copy()) for i in range(2)]  # Initialize bots

        self.washer = Animation()
        self.elevator = Animation()
    
    def _compile(self, events):
        time_unit = 50
        start_time = events[0].time - 5.0 / time_unit # Start after 1 second
        animations = []  # Sorted by start time
        bots = [None] * 2  # Two robots
        
        washer_start_time = None
        robot_update_time = None  # Updated by BedArrivedElevator / BedFinishedWashing / DropOffBed
        # This is done to compute duration for robot to travel to the event location
        
        _convert_time = lambda t: (t - start_time) * time_unit
        
        for event in events:
            match event:
                case BedArrivedElevator(time=t, elevator_id=elevator_id, buffer=buffer):
                    animations.append(ElevatorAnim(start=_convert_time(t), duration=1, elevator_id=elevator_id))
                    robot_update_time = t
                
                case BedArrivedWashing(time=t):
                    assert robot_update_time is None, "Robot update time should not be set"
                    robot_update_time = t

                    assert washer_start_time is None, "Washer should not be busy"
                
                case BedStartedWashing(time=t):
                    assert washer_start_time is None, "Washer should not be busy"
                    washer_start_time = _convert_time(t)

                case BedFinishedWashing(time=t):
                    robot_update_time = t

                    assert washer_start_time is not None, "Washer should be busy"
                    start = washer_start_time
                    duration = _convert_time(t) - start
                    animations.append(WasherAnim(start=start, duration=duration))
                    washer_start_time = None

                case PickUpBed(time=t, robot_id=bot_id, location_id=location_id, buffer=buffer):
                    assert robot_update_time is not None, "Robot update time should be set"
                    animations.append(BotAnim(start=_convert_time(t - robot_update_time), duration=1, bot_id=bot_id, location_id=location_id))
                    robot_update_time = None

                    assert bots[bot_id] is None, f"Bot should not be busy: {bot_id}"
                    bots[bot_id] = event
                
                case DropOffBed(time=t, robot_id=bot_id, location_id=location_id, buffer=buffer):
                    robot_update_time = t

                    assert bots[bot_id] is not None, f"Bot should be busy: {bot_id}"
                    
                    start = _convert_time(bots[bot_id].time)
                    duration = _convert_time(t) - start

                    animations.append(BotAnim(start=start, duration=duration, location_id=location_id, bot_id=bot_id))
                    bots[bot_id] = None
                

                
        
        # Add dummy event to let animations finish
        animations.append(ElevatorAnim(start=animations[-1].start + 10.0, duration=1, elevator_id=0))
        return animations

    def run(self):
        time = 5.0  
        anim = self.animations.pop(0)
        while True:
            self.handle_input()
            
            time += DT  # Trigger events based on time
            print(f"Time: {time:.2f} seconds", end="\r")
            if anim.start <= time:
                self.handle_anim(anim)
                anim = self.animations.pop(0) if self.animations else None

            self.step()
            self.draw(2)
            self.clock.tick(FPS)  # Limit to specified FPS
    
    def handle_anim(self, anim):
        animation_time = 1
        match anim:
            case ElevatorAnim(start=t, duration=d, elevator_id=elevator_id):
                print(f"\033[93m{type(anim).__name__}\033[0m at \033[92m{anim.start:.2f}\033[0m hours")
                self.elevator.start_animation(d)  # Start elevator animation
            
            case BotAnim(start=t, duration=d, bot_id=bot_id, location_id=location_id):
                print(f"\033[93m{type(anim).__name__}\033[0m at {anim.start:.2f} hours by Robot \033[92m{bot_id}\033[0m at Location \033[92m{location_id}\033[0m")
                self.bots[bot_id].start_animation(d, self.positions[location_id], 0)
            
            case WasherAnim(start=t, duration=d):
                print(f"\033[93m{type(anim).__name__}\033[0m at \033[92m{anim.start:.2f}\033[0m hours")
                self.washer.start_animation(d)  # Start washing animation
            
        
    def handle_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    quit()
    bot_goal = 1
    def step(self):
        for i, bot in enumerate(self.bots):
            bot.step()
        
        self.washer.step()
        self.elevator.step()


    def draw(self, data_point):
        self.screen.fill(BLACK)  # Clear the screen
        size = 50

        # Draw elevators top of the screen
        pg.draw.rect(self.screen, WHITE, (WIDTH // 2 - size // 2, HEIGHT // 8, size, size), border_radius=5)
        width = (1 - (1 - self.elevator.state) ** 1.5) * (size - 10)
        pg.draw.line(self.screen, BLACK, (WIDTH // 2, HEIGHT // 8), (WIDTH // 2, HEIGHT // 8 + size), int(width))

        # Draw robot positions
        for bot in self.bots:
            pg.draw.circle(self.screen, WHITE, (int(bot.pos.x), int(bot.pos.y)), 10)
            if bot.bed:
                pg.draw.rect(self.screen, RED if bot.bed == 1 else GREEN, (bot.pos.x - 8, bot.pos.y - 8, 16, 16), border_radius=5)
        
        # pg.draw.rect(self.screen, RED, (bed.x - 8, bed.y - 8, 16, 16), border_radius=5)

        # Washing machine
        pg.draw.rect(self.screen, WHITE, (WIDTH // 2 - size // 2, HEIGHT // 2 - size // 2, size, size), border_radius=10)
        pg.draw.circle(self.screen, BLACK, (WIDTH // 2, HEIGHT // 2), 18)
        r = (1 - (1 - self.washer.state) ** 1.5) * 18
        pg.draw.circle(self.screen, BLUE, (WIDTH // 2, HEIGHT // 2), r)
        
        pg.display.flip()




if __name__ == "__main__":
    with open("events.pkl", "rb") as f:
        events = pickle.load(f)
    sim = HospitalSimulation(events)
    sim.run()
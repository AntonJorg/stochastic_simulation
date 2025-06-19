import pygame as pg
from pygame.math import Vector2
import numpy as np

BLACK = (24, 24, 24)
WHITE = (200, 200, 200)
WIDTH, HEIGHT = 800, 600
MARGIN = 100
FLOORS = [HEIGHT // 3, 3 * HEIGHT // 4]
SIDES = [MARGIN, WIDTH - MARGIN - 50]

class HospitalSimulation:
    def __init__(self, fps=60):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Hospital Simulation")
        self.clock = pg.time.Clock()
        self.FPS = fps

        self.elevator = Elevator(MARGIN)
        self.bed = Bed(WIDTH - MARGIN - Bed.width)

        # States
        self.moving_elevator = True
        self.moving_bed = True
        self.loading_bed = False

    def run(self, data):
        # for data_point in data:
        while True:
            self.handle_input()
            self.step()
            self.draw(2)
            self.clock.tick(self.FPS)  # Limit to specified FPS
        
    def handle_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    quit()
    
    def step(self):
        if self.moving_elevator:
            self.moving_elevator = not self.elevator.move()
        
        if self.moving_bed:
            self.moving_bed = not self.bed.move()

        # Pickup bed if it is waiting and elevator is not moving
        elif not self.moving_elevator and not self.moving_bed and not self.loading_bed:
            self.elevator.beds.append(self.bed)
            self.bed.update_target(self.elevator.position.x)
            self.loading_bed = True
        
        if self.loading_bed:
            res = self.bed.move()

            if not res:
                self.moving_elevator = True  # Allow elevator to move after loading bed
                self.moving_bed = False  # Reset bed movement
                self.loading_bed = False
        state = np.array([self.moving_elevator, self.moving_bed, self.loading_bed], dtype=np.int8)
        print(state)
        
    def draw(self, data_point):
        self.screen.fill(BLACK)  # Clear the screen
        
        # Draw 2 floors of the hospital
        pg.draw.line(self.screen, WHITE, (MARGIN, FLOORS[0]), (WIDTH - MARGIN, FLOORS[0]), 2)
        pg.draw.line(self.screen, WHITE, (MARGIN, FLOORS[1]), (WIDTH - MARGIN, FLOORS[1]), 2)

        self.elevator.draw(self.screen)  # Draw the elevator
        self.bed.draw(self.screen)  # Draw the bed

        pg.display.flip()  # Update the display


class Elevator:
    width = 80
    height = 100
    speed = 0.05

    def __init__(self, x):
        self.position = Vector2(x - self.width, FLOORS[1])
        self.target = FLOORS[1] if self.position.y == FLOORS[0] else FLOORS[0]
        self.beds = []  # Move beds within the elevator
    

    def move(self):
        self.position.y += (self.target - self.position.y) * self.speed
        if abs(self.position.y - self.target) < 0.7:
            self.position.y = self.target
            self.target = FLOORS[0] if self.target == FLOORS[1] else FLOORS[1]
            return True  # Reached target, return True

        for bed in self.beds:
            bed.position.y = self.position.y - 6

        return False

    def draw(self, screen):
        # Draw a pair of cables offset from the center of the elevator
        cable_offset = 15
        center_x = self.position.x + self.width // 2
        pg.draw.line(screen, WHITE, (center_x - cable_offset, self.position.y - 97), (center_x - cable_offset, 0), 2)
        pg.draw.line(screen, WHITE, (center_x + cable_offset, self.position.y - 97), (center_x + cable_offset, 0), 2)
        
        # draw elevator
        pg.draw.rect(screen, WHITE, (self.position.x, self.position.y - 97, self.width, self.height), 2)

class Bed:
    width = 50
    height = 15
    speed = 5
    
    def __init__(self, x):
        self.position = Vector2(x, FLOORS[0])
        self.target = SIDES[0]
        self.dx = -self.speed

    def draw(self, screen):
        pg.draw.rect(screen, WHITE, (self.position.x, self.position.y - self.height - 6, self.width, self.height), 2)
        
        # draw two wheels
        wheel_radius = 5
        wheel_y = self.position.y - 6
        pg.draw.circle(screen, WHITE, (self.position.x + 5, wheel_y), wheel_radius)
        pg.draw.circle(screen, WHITE, (self.position.x + self.width - 5, wheel_y), wheel_radius)

    def move(self):
        self.position.x += self.dx

        if abs(self.position.x - self.target) < 1.0:
            self.position.x = self.target
            self.dx = -self.dx
            return True
        
        return False
        
    def update_target(self, target):
        self.target = target
        self.dx = self.speed if self.position.x < target else -self.speed

if __name__ == "__main__":
    data = np.cumsum(np.random.exponential(10, 100))  # Placeholder for actual data
    sim = HospitalSimulation()
    sim.run(data)
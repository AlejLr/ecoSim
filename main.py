from environment import Ecosystem
import pygame
import config


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_SIZE, config.WINDOW_SIZE))
    pygame.display.set_caption("Ecosystem Simulation")
    clock = pygame.time.Clock()

    env = Ecosystem(config.GRID_SIZE, config.TILE_SIZE)

    running = True
    while running:
        clock.tick(config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.update()
        env.render(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

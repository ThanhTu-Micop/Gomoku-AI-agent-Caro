import pygame
from src.game.constants import BOARD_SIZE, EMPTY, X, O


CELL_SIZE = 60
MARGIN = 40
GRID_COLOR = (139, 69, 19)
BG_COLOR = (222, 184, 135)
X_COLOR = (0, 0, 0)
O_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 0, 0, 128)
LINE_WIDTH = 2
STONE_RADIUS = 25


def get_screen_size() -> tuple[int, int]:
    size = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
    return size, size


def get_cell_pos(row: int, col: int) -> tuple[int, int]:
    x = MARGIN + col * CELL_SIZE
    y = MARGIN + row * CELL_SIZE
    return x, y


def draw_board(screen: pygame.Surface, grid, last_move: tuple[int, int] | None = None) -> None:
    screen.fill(BG_COLOR)

    for i in range(BOARD_SIZE):
        start = get_cell_pos(i, 0)
        end = get_cell_pos(i, BOARD_SIZE - 1)
        pygame.draw.line(screen, GRID_COLOR, start, end, LINE_WIDTH)

        start = get_cell_pos(0, i)
        end = get_cell_pos(BOARD_SIZE - 1, i)
        pygame.draw.line(screen, GRID_COLOR, start, end, LINE_WIDTH)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] != EMPTY:
                x, y = get_cell_pos(r, c)
                if grid[r, c] == X:
                    pygame.draw.circle(screen, X_COLOR, (x, y), STONE_RADIUS)
                else:
                    pygame.draw.circle(screen, O_COLOR, (x, y), STONE_RADIUS)
                    pygame.draw.circle(screen, X_COLOR, (x, y), STONE_RADIUS, 2)

    if last_move is not None:
        r, c = last_move
        x, y = get_cell_pos(r, c)
        highlight = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(highlight, HIGHLIGHT_COLOR, (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
        screen.blit(highlight, (x - STONE_RADIUS, y - STONE_RADIUS))


def draw_status(screen: pygame.Surface, message: str) -> None:
    font = pygame.font.SysFont(None, 32)
    text = font.render(message, True, X_COLOR)
    text_rect = text.get_rect(center=(get_screen_size()[0] // 2, get_screen_size()[1] - 20))
    screen.blit(text, text_rect)

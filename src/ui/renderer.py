import pygame
from src.game.constants import BOARD_SIZE, EMPTY, X, O


# --- Colour palette (inspired by modern web UI) ---
CELL_SIZE = 60
MARGIN = 40
LINE_WIDTH = 2
STONE_RADIUS = 25

# Board
BOARD_BG = (255, 255, 255)        # cell background
GRID_COLOR = (204, 204, 204)      # --line-color
BG_COLOR = (240, 242, 245)        # --bg-color window background

# Stones
X_COLOR = (255, 77, 79)           # --x-color  #ff4d4f
O_COLOR = (30, 224, 172)          # --o-color  #1ee0ac

# Effects
WIN_BG = (212, 237, 218)          # --win highlight  #d4edda
LAST_MOVE_COLOR = (255, 215, 0)   # gold ring for last move
STATUS_COLOR = (51, 51, 51)       # dark text


def get_screen_size() -> tuple[int, int]:
    size = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
    return size, size


def get_cell_pos(row: int, col: int) -> tuple[int, int]:
    x = MARGIN + col * CELL_SIZE
    y = MARGIN + row * CELL_SIZE
    return x, y


def draw_cell_bg(screen: pygame.Surface, row: int, col: int, colour: tuple[int, int, int]) -> None:
    x = MARGIN + col * CELL_SIZE
    y = MARGIN + row * CELL_SIZE
    half = CELL_SIZE // 2
    pygame.draw.rect(screen, colour, (x - half, y - half, CELL_SIZE, CELL_SIZE))


def draw_board(
    screen: pygame.Surface,
    grid,
    last_move: tuple[int, int] | None = None,
    win_cells: list[tuple[int, int]] | None = None,
) -> None:
    screen.fill(BG_COLOR)

    # Draw board background
    board_w = CELL_SIZE * (BOARD_SIZE - 1) + CELL_SIZE
    board_x = MARGIN - CELL_SIZE // 2
    board_y = MARGIN - CELL_SIZE // 2
    pygame.draw.rect(screen, BOARD_BG, (board_x, board_y, board_w, board_w))
    pygame.draw.rect(screen, GRID_COLOR, (board_x, board_y, board_w, board_w), 2)

    # Grid lines
    for i in range(BOARD_SIZE):
        start = get_cell_pos(i, 0)
        end = get_cell_pos(i, BOARD_SIZE - 1)
        pygame.draw.line(screen, GRID_COLOR, start, end, LINE_WIDTH)

        start = get_cell_pos(0, i)
        end = get_cell_pos(BOARD_SIZE - 1, i)
        pygame.draw.line(screen, GRID_COLOR, start, end, LINE_WIDTH)

    # Highlight winning cells
    win_set = set(win_cells) if win_cells else set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in win_set:
                draw_cell_bg(screen, r, c, WIN_BG)

    # Draw stones
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] != EMPTY:
                x, y = get_cell_pos(r, c)
                if grid[r, c] == X:
                    pygame.draw.circle(screen, X_COLOR, (x, y), STONE_RADIUS)
                    # X marker
                    off = STONE_RADIUS * 0.45
                    w = max(2, STONE_RADIUS // 8)
                    pygame.draw.line(screen, BOARD_BG, (x - off, y - off), (x + off, y + off), w)
                    pygame.draw.line(screen, BOARD_BG, (x + off, y - off), (x - off, y + off), w)
                else:
                    pygame.draw.circle(screen, O_COLOR, (x, y), STONE_RADIUS)

    # Last move ring
    if last_move is not None and last_move not in win_set:
        r, c = last_move
        x, y = get_cell_pos(r, c)
        pygame.draw.circle(screen, LAST_MOVE_COLOR, (x, y), STONE_RADIUS + 2, 3)


_status_font = None


def draw_status(screen: pygame.Surface, message: str) -> None:
    global _status_font
    if _status_font is None:
        _status_font = pygame.font.SysFont("segoeui", 28)
    text = _status_font.render(message, True, STATUS_COLOR)
    text_rect = text.get_rect(center=(get_screen_size()[0] // 2, get_screen_size()[1] + 20))
    screen.blit(text, text_rect)

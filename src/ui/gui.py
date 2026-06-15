import pygame
from src.game.board import Board
from src.game.rules import is_win, is_draw
from src.game.constants import X, O, BOARD_SIZE
from src.ui.renderer import draw_board, draw_status, get_screen_size, get_cell_pos, CELL_SIZE, MARGIN
from src.ai.base import Agent


AI_DELAY_MS = 500
WIN_COUNT = 5


def check_win_cells(grid, player: int, last_move: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = last_move
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        cells = [(r0, c0)]
        for sign in (1, -1):
            r, c = r0 + sign * dr, c0 + sign * dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
                cells.append((r, c))
                r += sign * dr
                c += sign * dc
        if len(cells) >= WIN_COUNT:
            return cells
    return None


class HumanAgent(Agent):
    def get_move(self, grid, player):
        return None


def main(
    player1: Agent | None = None,
    player2: Agent | None = None,
) -> None:
    pygame.init()
    screen_size = get_screen_size()
    screen = pygame.display.set_mode((screen_size[0], screen_size[1] + 50))
    pygame.display.set_caption("Cờ Caro 9x9")
    clock = pygame.time.Clock()

    board = Board()
    current_player = X
    game_over = False
    winner = None
    last_move = None
    win_cells: list[tuple[int, int]] = []

    human_agent = HumanAgent()
    p1 = player1 if player1 is not None else human_agent
    p2 = player2 if player2 is not None else human_agent

    ai_thinking = False
    ai_move_time = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                if isinstance(p1 if current_player == X else p2, HumanAgent):
                    x, y = event.pos
                    col = int((x - MARGIN + CELL_SIZE // 2) / CELL_SIZE)
                    row = int((y - MARGIN + CELL_SIZE // 2) / CELL_SIZE)
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board.is_empty(row, col):
                        board.place(row, col, current_player)
                        last_move = (row, col)
                        if is_win(board.grid, current_player, last_move=(row, col)):
                            game_over = True
                            winner = current_player
                            win_cells = check_win_cells(board.grid, current_player, last_move) or []
                        elif is_draw(board.grid):
                            game_over = True
                            winner = None
                        else:
                            current_player = O if current_player == X else X
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    board.reset()
                    current_player = X
                    game_over = False
                    winner = None
                    last_move = None
                    win_cells = []
                    ai_thinking = False

        if not game_over:
            agent = p1 if current_player == X else p2
            if not isinstance(agent, HumanAgent) and not ai_thinking:
                ai_thinking = True
                ai_move_time = pygame.time.get_ticks()

            if not isinstance(agent, HumanAgent) and ai_thinking:
                now = pygame.time.get_ticks()
                if now - ai_move_time > AI_DELAY_MS:
                    move = agent.get_move(board.grid.copy(), current_player)
                    if move is not None:
                        r, c = move
                        board.place(r, c, current_player)
                        last_move = (r, c)
                        if is_win(board.grid, current_player, last_move=(r, c)):
                            game_over = True
                            winner = current_player
                            win_cells = check_win_cells(board.grid, current_player, last_move) or []
                        elif is_draw(board.grid):
                            game_over = True
                            winner = None
                        else:
                            current_player = O if current_player == X else X
                    ai_thinking = False

        draw_board(screen, board.grid, last_move, win_cells)

        name = "X" if current_player == X else "O"
        if game_over:
            if winner is None:
                draw_status(screen, "Hòa! Nhấn R để chơi lại")
            else:
                w_name = "X" if winner == X else "O"
                draw_status(screen, f"{w_name} thắng! Nhấn R để chơi lại")
        else:
            agent = p1 if current_player == X else p2
            if isinstance(agent, HumanAgent):
                draw_status(screen, f"Lượt {name} (Người)")
            else:
                draw_status(screen, f"Lượt {name} (AI)")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

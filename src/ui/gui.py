import pygame
import os
from src.game.board import Board
from src.game.rules import is_win, is_draw
from src.game.constants import X, O, BOARD_SIZE, EMPTY
from src.ui.renderer import draw_board, draw_status, get_screen_size, draw_popup_modal, UIButton, CELL_SIZE, MARGIN
from src.ai.base import Agent
from src.ai.minimax import MinimaxAgent
from src.ai.rl_agent import AlphaZeroAgent

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

def create_agent(ai_type: str, param: int) -> Agent:
    """Khởi tạo AI Agent và gắn nhãn để UI có thể đọc thông tin khi Game Over"""
    if ai_type == "Minimax":
        agent = MinimaxAgent(depth=param)
        agent.gui_name = "Minimax"
        agent.gui_param = param
        return agent
    else:
        agent = AlphaZeroAgent(num_simulations=param)
        if os.path.exists("models/rl_agent.pth"):
            try:
                agent.load("models/rl_agent.pth")
            except Exception as e:
                print(f"Lỗi khi tải trọng số AlphaZero: {e}")
        agent.gui_name = "AlphaZero"
        agent.gui_param = param
        return agent

def adjust_param(ai_type: str, param: int, action: str) -> int:
    """Hàm hỗ trợ logic tăng/giảm nút bấm"""
    if ai_type == "Minimax":
        if action == "dec" and param > 0: return param - 1
        if action == "inc": return param + 1
    elif ai_type == "AlphaZero":
        if action == "dec" and param >= 20: return param - 20
        if action == "inc" and param < 800: return param + 20
    return param

def main(
    player1: Agent | None = None,
    player2: Agent | None = None,
) -> None:
    pygame.init()
    width, height = get_screen_size()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Gomoku 9x9 - AI Agent Configuration")
    clock = pygame.time.Clock()

    panel_w, panel_h = 420, 240
    panel_x = (width - panel_w) // 2
    panel_y = (width - panel_h) // 2
    
    # --- MENU CHỌN CHẾ ĐỘ CHƠI ---
    btn_mode_hvh = UIButton(width // 2 - 150, panel_y + 110, 300, 35, "Human vs Human", (155, 89, 182))
    btn_mode_hvai = UIButton(width // 2 - 150, panel_y + 155, 145, 35, "Human vs AI", (52, 152, 219))
    btn_mode_aivai = UIButton(width // 2 + 5, panel_y + 155, 145, 35, "AI vs AI", (46, 204, 113))

    # --- MENU CẤU HÌNH NGƯỜI VS AI ---
    btn_side_x = UIButton(panel_x + 20, panel_y + 115, 85, 30, "Side: X", (255, 77, 79))
    btn_side_o = UIButton(panel_x + 110, panel_y + 115, 85, 30, "Side: O", (30, 224, 172))
    btn_ai_minimax = UIButton(panel_x + 225, panel_y + 115, 85, 30, "Minimax", (44, 62, 80))
    btn_ai_az = UIButton(panel_x + 315, panel_y + 115, 85, 30, "AlphaZero", (127, 140, 141))
    
    btn_param_dec = UIButton(panel_x + 20, panel_y + 155, 40, 30, "-", (149, 165, 166))
    btn_param_val = UIButton(panel_x + 70, panel_y + 155, 280, 30, "Depth: 3", (236, 240, 241), (44, 62, 80))
    btn_param_inc = UIButton(panel_x + 360, panel_y + 155, 40, 30, "+", (149, 165, 166))
    btn_back = UIButton(panel_x + 20, panel_y + 195, 90, 35, "Back", (192, 57, 43))
    btn_start = UIButton(panel_x + 120, panel_y + 195, 280, 35, "START MATCH", (39, 174, 96))

    # --- MENU CẤU HÌNH AI VS AI (PLAYER X) ---
    btn_p1_minimax = UIButton(panel_x + 20, panel_y + 115, 185, 30, "Minimax", (44, 62, 80))
    btn_p1_az = UIButton(panel_x + 215, panel_y + 115, 185, 30, "AlphaZero", (127, 140, 141))
    btn_p1_param_dec = UIButton(panel_x + 20, panel_y + 155, 40, 30, "-", (149, 165, 166))
    btn_p1_param_val = UIButton(panel_x + 70, panel_y + 155, 280, 30, "Depth: 3", (236, 240, 241), (44, 62, 80))
    btn_p1_param_inc = UIButton(panel_x + 360, panel_y + 155, 40, 30, "+", (149, 165, 166))
    btn_p1_back = UIButton(panel_x + 20, panel_y + 195, 90, 35, "Back", (192, 57, 43))
    btn_p1_next = UIButton(panel_x + 120, panel_y + 195, 280, 35, "Next (Config O)", (39, 174, 96))

    # --- MENU CẤU HÌNH AI VS AI (PLAYER O) ---
    btn_p2_minimax = UIButton(panel_x + 20, panel_y + 115, 185, 30, "Minimax", (44, 62, 80))
    btn_p2_az = UIButton(panel_x + 215, panel_y + 115, 185, 30, "AlphaZero", (127, 140, 141))
    btn_p2_param_dec = UIButton(panel_x + 20, panel_y + 155, 40, 30, "-", (149, 165, 166))
    btn_p2_param_val = UIButton(panel_x + 70, panel_y + 155, 280, 30, "Depth: 3", (236, 240, 241), (44, 62, 80))
    btn_p2_param_inc = UIButton(panel_x + 360, panel_y + 155, 40, 30, "+", (149, 165, 166))
    btn_p2_back = UIButton(panel_x + 20, panel_y + 195, 90, 35, "Back", (192, 57, 43))
    btn_p2_start = UIButton(panel_x + 120, panel_y + 195, 280, 35, "START AI VS AI", (39, 174, 96))

    # --- NÚT ĐI LẠI VÀ CHƠI LẠI TRONG TRẬN ---
    btn_restart = UIButton(width // 2 - 80, panel_y + 145, 160, 50, "Play Again", (52, 152, 219))
    btn_undo = UIButton(width - 120, height - 60, 100, 40, "Undo", (243, 156, 18))

    game_state = "MENU_MODE"

    # Cấu hình AI động
    human_side = X
    ai_type_hvai, ai_param_hvai = "Minimax", 3
    ai_type_p1, ai_param_p1 = "Minimax", 3
    ai_type_p2, ai_param_p2 = "AlphaZero", 80
    
    # Biến lưu trữ chuỗi bàn phím gõ
    input_buffer = ""

    human_agent = HumanAgent()
    p1, p2 = human_agent, human_agent
    board = Board()
    current_player = X
    game_over = False
    winner = None
    last_move = None
    win_cells = []
    move_history = []
    ai_thinking = False
    ai_move_time = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # --- CƠ CHẾ NHẬP SỐ TỪ BÀN PHÍM ---
            elif event.type == pygame.KEYDOWN:
                if game_state in ["MENU_HVAI_CONFIG", "MENU_AIVAI_CONFIG_P1", "MENU_AIVAI_CONFIG_P2"]:
                    if event.key == pygame.K_BACKSPACE:
                        input_buffer = input_buffer[:-1]
                    elif event.unicode.isdigit():
                        if len(input_buffer) < 4:  # Giới hạn độ dài để tránh số quá lớn gây crash
                            input_buffer += event.unicode
                    
                    val = int(input_buffer) if input_buffer else 0
                    if game_state == "MENU_HVAI_CONFIG": ai_param_hvai = val
                    elif game_state == "MENU_AIVAI_CONFIG_P1": ai_param_p1 = val
                    elif game_state == "MENU_AIVAI_CONFIG_P2": ai_param_p2 = val
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos

                # --- 1. XỬ LÝ CLICK: MENU CHỌN CHẾ ĐỘ ---
                if game_state == "MENU_MODE":
                    if btn_mode_hvh.is_clicked(pos):
                        p1, p2 = human_agent, human_agent
                        game_state = "PLAYING"
                    elif btn_mode_hvai.is_clicked(pos):
                        input_buffer = str(ai_param_hvai)
                        game_state = "MENU_HVAI_CONFIG"
                    elif btn_mode_aivai.is_clicked(pos):
                        input_buffer = str(ai_param_p1)
                        game_state = "MENU_AIVAI_CONFIG_P1"

                # --- 2. XỬ LÝ CLICK: CẤU HÌNH NGƯỜI VS AI ---
                elif game_state == "MENU_HVAI_CONFIG":
                    if btn_side_x.is_clicked(pos): human_side = X
                    elif btn_side_o.is_clicked(pos): human_side = O
                    elif btn_ai_minimax.is_clicked(pos):
                        ai_type_hvai, ai_param_hvai = "Minimax", 3
                        input_buffer = str(ai_param_hvai)
                    elif btn_ai_az.is_clicked(pos):
                        ai_type_hvai, ai_param_hvai = "AlphaZero", 80
                        input_buffer = str(ai_param_hvai)
                    elif btn_param_dec.is_clicked(pos):
                        ai_param_hvai = adjust_param(ai_type_hvai, ai_param_hvai, "dec")
                        input_buffer = str(ai_param_hvai)
                    elif btn_param_inc.is_clicked(pos):
                        ai_param_hvai = adjust_param(ai_type_hvai, ai_param_hvai, "inc")
                        input_buffer = str(ai_param_hvai)
                    elif btn_back.is_clicked(pos):
                        game_state = "MENU_MODE"
                    elif btn_start.is_clicked(pos):
                        if human_side == X:
                            p1, p2 = human_agent, create_agent(ai_type_hvai, ai_param_hvai)
                        else:
                            p1, p2 = create_agent(ai_type_hvai, ai_param_hvai), human_agent
                        game_state = "PLAYING"

                # --- 3. XỬ LÝ CLICK: AI VS AI (PLAYER X) ---
                elif game_state == "MENU_AIVAI_CONFIG_P1":
                    if btn_p1_minimax.is_clicked(pos):
                        ai_type_p1, ai_param_p1 = "Minimax", 3
                        input_buffer = str(ai_param_p1)
                    elif btn_p1_az.is_clicked(pos):
                        ai_type_p1, ai_param_p1 = "AlphaZero", 80
                        input_buffer = str(ai_param_p1)
                    elif btn_p1_param_dec.is_clicked(pos):
                        ai_param_p1 = adjust_param(ai_type_p1, ai_param_p1, "dec")
                        input_buffer = str(ai_param_p1)
                    elif btn_p1_param_inc.is_clicked(pos):
                        ai_param_p1 = adjust_param(ai_type_p1, ai_param_p1, "inc")
                        input_buffer = str(ai_param_p1)
                    elif btn_p1_back.is_clicked(pos):
                        game_state = "MENU_MODE"
                    elif btn_p1_next.is_clicked(pos):
                        input_buffer = str(ai_param_p2)
                        game_state = "MENU_AIVAI_CONFIG_P2"

                # --- 4. XỬ LÝ CLICK: AI VS AI (PLAYER O) ---
                elif game_state == "MENU_AIVAI_CONFIG_P2":
                    if btn_p2_minimax.is_clicked(pos):
                        ai_type_p2, ai_param_p2 = "Minimax", 3
                        input_buffer = str(ai_param_p2)
                    elif btn_p2_az.is_clicked(pos):
                        ai_type_p2, ai_param_p2 = "AlphaZero", 80
                        input_buffer = str(ai_param_p2)
                    elif btn_p2_param_dec.is_clicked(pos):
                        ai_param_p2 = adjust_param(ai_type_p2, ai_param_p2, "dec")
                        input_buffer = str(ai_param_p2)
                    elif btn_p2_param_inc.is_clicked(pos):
                        ai_param_p2 = adjust_param(ai_type_p2, ai_param_p2, "inc")
                        input_buffer = str(ai_param_p2)
                    elif btn_p2_back.is_clicked(pos):
                        input_buffer = str(ai_param_p1)
                        game_state = "MENU_AIVAI_CONFIG_P1"
                    elif btn_p2_start.is_clicked(pos):
                        p1 = create_agent(ai_type_p1, ai_param_p1)
                        p2 = create_agent(ai_type_p2, ai_param_p2)
                        game_state = "PLAYING"

                # --- 5. XỬ LÝ CLICK: KHI GAME OVER ---
                elif game_state == "GAME_OVER":
                    if btn_restart.is_clicked(pos):
                        board.reset()
                        current_player = X
                        game_over = False
                        winner = None
                        last_move = None
                        win_cells = []
                        move_history = []
                        ai_thinking = False
                        game_state = "MENU_MODE"

                # --- 6. CLICK ĐẶT QUÂN CỜ TRONG TRẬN ---
                elif game_state == "PLAYING" and not game_over:
                    if btn_undo.is_clicked(pos) and len(move_history) > 0 and not ai_thinking:
                        is_vs_ai = not isinstance(p1, HumanAgent) or not isinstance(p2, HumanAgent)
                        pops = 2 if is_vs_ai and len(move_history) >= 2 else 1
                        
                        if is_vs_ai and len(move_history) == 1 and not isinstance(p1, HumanAgent):
                            pass 
                        else:
                            for _ in range(pops):
                                r, c, p = move_history.pop()
                                board.grid[r, c] = EMPTY
                            current_player = X if len(move_history) % 2 == 0 else O
                            last_move = (move_history[-1][0], move_history[-1][1]) if move_history else None
                        continue 

                    agent = p1 if current_player == X else p2
                    if isinstance(agent, HumanAgent):
                        x, y = pos
                        board_limit = MARGIN + BOARD_SIZE * CELL_SIZE
                        if MARGIN <= x <= board_limit and MARGIN <= y <= board_limit:
                            col = (x - MARGIN) // CELL_SIZE
                            row = (y - MARGIN) // CELL_SIZE
                            
                            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board.is_empty(row, col):
                                board.place(row, col, current_player)
                                last_move = (row, col)
                                move_history.append((row, col, current_player))
                                
                                if is_win(board.grid, current_player, last_move=(row, col)):
                                    game_over = True
                                    winner = current_player
                                    win_cells = check_win_cells(board.grid, current_player, last_move) or []
                                    game_state = "GAME_OVER"
                                elif is_draw(board.grid):
                                    game_over = True
                                    winner = None
                                    game_state = "GAME_OVER"
                                else:
                                    current_player = O if current_player == X else X

        # --- XỬ LÝ LƯỢT AI TỰ ĐỘNG ---
        if game_state == "PLAYING" and not game_over:
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
                        move_history.append((r, c, current_player))
                        
                        if is_win(board.grid, current_player, last_move=(r, c)):
                            game_over = True
                            winner = current_player
                            win_cells = check_win_cells(board.grid, current_player, last_move) or []
                            game_state = "GAME_OVER"
                        elif is_draw(board.grid):
                            game_over = True
                            winner = None
                            game_state = "GAME_OVER"
                        else:
                            current_player = O if current_player == X else X
                    ai_thinking = False

        # =========================================================
        # RENDER GIAO DIỆN VÀ POPUP MODAL THEO STATE
        # =========================================================
        draw_board(screen, board.grid, last_move, win_cells)

        if not game_over:
            name = "X" if current_player == X else "O"
            agent = p1 if current_player == X else p2
            if isinstance(agent, HumanAgent):
                draw_status(screen, f"Turn: P{current_player} ({name} - Human)")
            else:
                ai_name = getattr(agent, 'gui_name', "AI")
                draw_status(screen, f"Turn: P{current_player} ({name} - {ai_name} thinking...)")
                
            if len(move_history) > 0 and not ai_thinking and game_state == "PLAYING":
                btn_undo.draw(screen)
        else:
            draw_status(screen, "Match Finished!")

        # Nhấp nháy con trỏ khi nhập liệu
        cursor = "_" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""

        if game_state == "MENU_MODE":
            draw_popup_modal(screen, "GOMOKU GAME", "Select game mode to play", [btn_mode_hvh, btn_mode_hvai, btn_mode_aivai])
            
        elif game_state == "MENU_HVAI_CONFIG":
            btn_side_x.bg_color = (255, 77, 79) if human_side == X else (127, 140, 141)
            btn_side_o.bg_color = (30, 224, 172) if human_side == O else (127, 140, 141)
            btn_ai_minimax.bg_color = (41, 128, 185) if ai_type_hvai == "Minimax" else (127, 140, 141)
            btn_ai_az.bg_color = (155, 89, 182) if ai_type_hvai == "AlphaZero" else (127, 140, 141)
            
            label = "Depth" if ai_type_hvai == "Minimax" else "Sims"
            btn_param_val.text = f"{label}: {input_buffer}{cursor}"
            
            draw_popup_modal(screen, "HUMAN VS AI", "Configure player side & AI engine", 
                             [btn_side_x, btn_side_o, btn_ai_minimax, btn_ai_az, btn_param_dec, btn_param_val, btn_param_inc, btn_back, btn_start])
                             
        elif game_state == "MENU_AIVAI_CONFIG_P1":
            btn_p1_minimax.bg_color = (41, 128, 185) if ai_type_p1 == "Minimax" else (127, 140, 141)
            btn_p1_az.bg_color = (155, 89, 182) if ai_type_p1 == "AlphaZero" else (127, 140, 141)
            
            label = "Depth" if ai_type_p1 == "Minimax" else "Sims"
            btn_p1_param_val.text = f"{label}: {input_buffer}{cursor}"
            
            draw_popup_modal(screen, "PLAYER X CONFIG", "Select AI algorithm for Player X", 
                             [btn_p1_minimax, btn_p1_az, btn_p1_param_dec, btn_p1_param_val, btn_p1_param_inc, btn_p1_back, btn_p1_next])
                             
        elif game_state == "MENU_AIVAI_CONFIG_P2":
            btn_p2_minimax.bg_color = (41, 128, 185) if ai_type_p2 == "Minimax" else (127, 140, 141)
            btn_p2_az.bg_color = (155, 89, 182) if ai_type_p2 == "AlphaZero" else (127, 140, 141)
            
            label = "Depth" if ai_type_p2 == "Minimax" else "Sims"
            btn_p2_param_val.text = f"{label}: {input_buffer}{cursor}"
            
            draw_popup_modal(screen, "PLAYER O CONFIG", "Select AI algorithm for Player O", 
                             [btn_p2_minimax, btn_p2_az, btn_p2_param_dec, btn_p2_param_val, btn_p2_param_inc, btn_p2_back, btn_p2_start])

        elif game_state == "GAME_OVER":
            if winner is None:
                draw_popup_modal(screen, "DRAW", "The board is full. No more valid moves!", [btn_restart])
            else:
                w_name = "X" if winner == X else "O"
                winning_agent = p1 if winner == X else p2
                
                if isinstance(winning_agent, HumanAgent):
                    draw_popup_modal(screen, f"PLAYER {w_name} WINS!", "Congratulations! Human takes the victory.", [btn_restart])
                else:
                    ai_name = getattr(winning_agent, 'gui_name', "AI")
                    ai_param = getattr(winning_agent, 'gui_param', 0)
                    param_lbl = "Depth" if ai_name == "Minimax" else "Sims"
                    
                    draw_popup_modal(screen, f"{ai_name} ({param_lbl}: {ai_param}) WINS!", f"Player {w_name} dominated the match.", [btn_restart])

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
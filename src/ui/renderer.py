import os
import pygame
from src.game.constants import BOARD_SIZE, EMPTY, X, O


# --- Colour palette (inspired by modern web UI) ---
CELL_SIZE = 60
MARGIN = 40
LINE_WIDTH = 2
STONE_RADIUS = 25

# Board
BOARD_BG = (255, 255, 255)
GRID_COLOR = (204, 204, 204)
BG_COLOR = (240, 242, 245)

# Stones
X_COLOR = (255, 77, 79)
O_COLOR = (30, 224, 172)

# Effects
WIN_BG = (212, 237, 218)
LAST_MOVE_COLOR = (255, 215, 0)
STATUS_COLOR = (51, 51, 51)


# ==========================================
# LỚP NÚT BẤM PHẲNG (BUTTON UI)
# ==========================================
class UIButton:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, bg_color: tuple[int, int, int], text_color: tuple[int, int, int] = (255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.bg_color = bg_color
        self.text_color = text_color
        self.font = pygame.font.SysFont("segoeui", 20, bold=True)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=8)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)


# ==========================================
# CÁC HÀM VẼ GIAO DIỆN BÀN CỜ
# ==========================================
def get_screen_size() -> tuple[int, int]:
    board_w = CELL_SIZE * BOARD_SIZE + MARGIN * 2
    return board_w, board_w + 80

def get_cell_pos(row: int, col: int) -> tuple[int, int]:
    x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
    y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
    return x, y

def draw_cell_bg(screen: pygame.Surface, row: int, col: int, colour: tuple[int, int, int]) -> None:
    x = MARGIN + col * CELL_SIZE
    y = MARGIN + row * CELL_SIZE
    pygame.draw.rect(screen, colour, (x, y, CELL_SIZE, CELL_SIZE))

# --- BIẾN TOÀN CỤC LƯU ẢNH (CACHE) ---
_img_x = None
_img_o = None

def _load_piece_images():
    """Hàm tải 2 file ảnh X và O tách rời"""
    global _img_x, _img_o
    if _img_x is None and _img_o is None:
        piece_size = STONE_RADIUS * 2
        
        if os.path.exists("assets/x.png") and os.path.exists("assets/o.png"):
            try:
                # Tải thẳng từng file ảnh độc lập
                x_raw = pygame.image.load("assets/x.png").convert_alpha()
                _img_x = pygame.transform.smoothscale(x_raw, (piece_size, piece_size))
                
                o_raw = pygame.image.load("assets/o.png").convert_alpha()
                _img_o = pygame.transform.smoothscale(o_raw, (piece_size, piece_size))
                
            except Exception as e:
                print(f"Lỗi khi tải ảnh: {e}")
                _img_x = "ERROR"
                _img_o = "ERROR"
        else:
            _img_x = "NOT_FOUND"
            _img_o = "NOT_FOUND"


def draw_board(
    screen: pygame.Surface,
    grid,
    last_move: tuple[int, int] | None = None,
    win_cells: list[tuple[int, int]] | None = None,
) -> None:
    screen.fill(BG_COLOR)

    # 1. Vẽ khung viền nền của toàn bộ bảng cờ
    board_w = CELL_SIZE * BOARD_SIZE
    pygame.draw.rect(screen, BOARD_BG, (MARGIN, MARGIN, board_w, board_w))
    pygame.draw.rect(screen, GRID_COLOR, (MARGIN, MARGIN, board_w, board_w), LINE_WIDTH)

    # 2. Vẽ lưới tạo thành các ô vuông
    for i in range(1, BOARD_SIZE):
        start_x = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, GRID_COLOR, (start_x, MARGIN), (start_x, MARGIN + board_w), LINE_WIDTH)
        
        start_y = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, GRID_COLOR, (MARGIN, start_y), (MARGIN + board_w, start_y), LINE_WIDTH)

    # 3. Highlight nền xanh cho các ô chiến thắng
    win_set = set(win_cells) if win_cells else set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in win_set:
                draw_cell_bg(screen, r, c, WIN_BG)

    # 4. Tải ảnh (Chỉ chạy thực sự ở vòng lặp đầu tiên)
    _load_piece_images()

    # 5. Vẽ các quân cờ đã đánh
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] != EMPTY:
                x, y = get_cell_pos(r, c)
                
                if grid[r, c] == X:
                    # Nếu tải ảnh thành công thì in ảnh, nếu lỗi thì vẽ hình cũ
                    if isinstance(_img_x, pygame.Surface):
                        rect = _img_x.get_rect(center=(x, y))
                        screen.blit(_img_x, rect)
                    else:
                        pygame.draw.circle(screen, X_COLOR, (x, y), STONE_RADIUS)
                        off = STONE_RADIUS * 0.45
                        w = max(2, STONE_RADIUS // 8)
                        pygame.draw.line(screen, BOARD_BG, (x - off, y - off), (x + off, y + off), w)
                        pygame.draw.line(screen, BOARD_BG, (x + off, y - off), (x - off, y + off), w)
                
                else: # Đánh O
                    if isinstance(_img_o, pygame.Surface):
                        rect = _img_o.get_rect(center=(x, y))
                        screen.blit(_img_o, rect)
                    else:
                        pygame.draw.circle(screen, O_COLOR, (x, y), STONE_RADIUS)

    # 6. Hiệu ứng vòng tròn viền vàng bọc ngoài đánh dấu nước đi cuối cùng
    if last_move is not None and last_move not in win_set:
        r, c = last_move
        x, y = get_cell_pos(r, c)
        
        # Nếu dùng ảnh thì vòng highlight cần rộng hơn chút xíu để không che mất ảnh
        highlight_radius = STONE_RADIUS + 4 if isinstance(_img_x, pygame.Surface) else STONE_RADIUS + 2
        pygame.draw.circle(screen, LAST_MOVE_COLOR, (x, y), highlight_radius, 3)


_status_font = None

def draw_status(screen: pygame.Surface, message: str) -> None:
    global _status_font
    if _status_font is None:
        _status_font = pygame.font.SysFont("segoeui", 22, bold=True)
    text = _status_font.render(message, True, STATUS_COLOR)
    w, h = get_screen_size()
    text_rect = text.get_rect(center=(w // 2, h - 40))
    screen.blit(text, text_rect)


# ==========================================
# HÀM VẼ KHUNG THÔNG BÁO POPUP Ở GIỮA MÀN HÌNH
# ==========================================
def draw_popup_modal(screen: pygame.Surface, title: str, subtitle: str, buttons: list[UIButton]) -> None:
    w, h = get_screen_size()
    
    shadow = pygame.Surface((w, h), pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 150))
    screen.blit(shadow, (0, 0))
    
    panel_w, panel_h = 420, 240
    panel_x = (w - panel_w) // 2
    panel_y = (w - panel_h) // 2
    
    pygame.draw.rect(screen, (255, 255, 255), (panel_x, panel_y, panel_w, panel_h), border_radius=12)
    pygame.draw.rect(screen, (180, 190, 200), (panel_x, panel_y, panel_w, panel_h), width=2, border_radius=12)
    
    if "VICTORY" in title or "WINS" in title:
        title_color = (46, 204, 113)
    elif "DEFEAT" in title:
        title_color = (231, 76, 60)
    else:
        title_color = (44, 62, 80)
        
    font_title = pygame.font.SysFont("segoeui", 34, bold=True)
    title_surf = font_title.render(title, True, title_color)
    title_rect = title_surf.get_rect(center=(w // 2, panel_y + 55))
    screen.blit(title_surf, title_rect)
    
    if subtitle:
        font_sub = pygame.font.SysFont("segoeui", 18)
        sub_surf = font_sub.render(subtitle, True, (127, 140, 141))
        sub_rect = sub_surf.get_rect(center=(w // 2, panel_y + 105))
        screen.blit(sub_surf, sub_rect)
        
    for btn in buttons:
        btn.draw(screen)
import pygame
import sys
import constants
import tetris
class ScreenManager:
    def __init__(self, screen_width, screen_height, x, y):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.x = x
        self.y = y
        pygame.display.set_caption("Menu Game")

    def draw_text(self, text, font, color, x, y):
        text_obj = font.render(text, True, color)
        text_rect = text_obj.get_rect()
        text_rect.center = (x, y)
        self.screen.blit(text_obj, text_rect)

    def draw_menu(self):
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        font = pygame.font.Font(None, 36)

        while True:
            self.screen.fill(WHITE)
            self.draw_text('Menu Game', font, BLACK, self.screen_width // 2, 100)

            # Vẽ các lựa chọn
            play_button = pygame.Rect(self.screen_width // 2 - 100, 200, 200, 50)
            self.draw_text('Play', font, BLACK, self.screen_width // 2, 225)
            pygame.draw.rect(self.screen, BLACK, play_button, 2)

            instructions_button = pygame.Rect(self.screen_width // 2 - 100, 300, 200, 50)
            self.draw_text('Tutorial', font, BLACK, self.screen_width // 2, 325)
            pygame.draw.rect(self.screen, BLACK, instructions_button, 2)

            quit_button = pygame.Rect(self.screen_width // 2 - 100, 400, 200, 50)
            self.draw_text('Quit', font, BLACK, self.screen_width // 2, 425)
            pygame.draw.rect(self.screen, BLACK, quit_button, 2)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
             
                    # Kiểm tra xem chuột ở đâu
                    if play_button.collidepoint(mouse_pos):
                        # Thực hiện hành động khi nhấp vào nút "Chơi"
                        # Đây là nơi bạn gọi hàm để chuyển sang màn hình chơi game
                        tetris.Run_game(self.x, self.y)
                        pass
                    elif instructions_button.collidepoint(mouse_pos):
                        # Thực hiện hành động khi nhấp vào nút "Hướng dẫn"
                        print("Hiển thị hướng dẫn!")
                    elif quit_button.collidepoint(mouse_pos):
                        # Thực hiện hành động khi nhấp vào nút "Thoát"
                        pygame.quit()
                        sys.exit()

            pygame.display.update()

# Sử dụng
def Run_game(x, y):
    pygame.init()
    screen_width = x*constants.BWIDTH+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
    screen_height = y*constants.BHEIGHT+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
    screen_manager = ScreenManager(screen_width, screen_height, x, y)
    screen_manager.draw_menu()

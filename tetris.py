#!/usr/bin/env python3
import asyncio
import threading
import sys
# File: tetris.py 
# Description: Main file with tetris game.
# Author: Pavel Benáček <pavel.benacek@gmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pygame
import pdb

import random
import math
import block
import constants

import main_app 
a, b = 'a', 'a'

class Tetris(object):
    """
    The class with implementation of tetris game logic.
    """

    def __init__(self,bx,by):
        """
        Initialize the tetris object.

        Parameters:
            - bx - number of blocks in x
            - by - number of blocks in y
        """
        self.bx = bx
        self.by = by
        # Compute the resolution of the play board based on the required number of blocks.
        self.resx = bx*constants.BWIDTH+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        self.resy = by*constants.BHEIGHT+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        # Prepare the pygame board objects (white lines)
        self.board_up    = pygame.Rect(0,constants.BOARD_UP_MARGIN,self.resx,constants.BOARD_HEIGHT)
        self.board_down  = pygame.Rect(0,self.resy-constants.BOARD_HEIGHT,self.resx,constants.BOARD_HEIGHT)
        self.board_left  = pygame.Rect(0,constants.BOARD_UP_MARGIN,constants.BOARD_HEIGHT,self.resy)
        self.board_right = pygame.Rect(self.resx-constants.BOARD_HEIGHT,constants.BOARD_UP_MARGIN,constants.BOARD_HEIGHT,self.resy)
        # List of used blocks
        self.blk_list    = []
        # Compute start indexes for tetris blocks
        self.start_x = math.ceil(self.resx/2.0)
        self.start_y = constants.BOARD_UP_MARGIN + constants.BOARD_HEIGHT + constants.BOARD_MARGIN
        # Blocka data (shapes and colors). The shape is encoded in the list of [X,Y] points. Each point
        # represents the relative position. The true/false value is used for the configuration of rotation where
        # False means no rotate and True allows the rotation.
        self.block_data = (
            ([[0,0],[1,0],[2,0],[3,0]],constants.RED,True),     # I block 
            ([[0,0],[1,0],[0,1],[-1,1]],constants.GREEN,True),  # S block 
            ([[0,0],[1,0],[2,0],[2,1]],constants.BLUE,True),    # J block
            ([[0,0],[0,1],[1,0],[1,1]],constants.ORANGE,False), # O block
            ([[-1,0],[0,0],[0,1],[1,1]],constants.GOLD,True),   # Z block
            ([[0,0],[1,0],[2,0],[1,1]],constants.PURPLE,True),  # T block
            ([[0,0],[1,0],[2,0],[0,1]],constants.CYAN,True),    # J block
        )
        # Compute the number of blocks. When the number of blocks is even, we can use it directly but 
        # we have to decrese the number of blocks in line by one when the number is odd (because of the used margin).
        self.blocks_in_line = bx if bx%2 == 0 else bx-1
        self.blocks_in_pile = by
        # Score settings
        self.score = 0
        # Remember the current speed 
        self.speed = 1
        # The score level threshold
        self.score_level = constants.SCORE_LEVEL
    def draw_text(self, text, font, color, x, y, font_size):
        """
        Vẽ văn bản lên màn hình.

        Parameters:
            - text: Văn bản cần vẽ.
            - font: Phông chữ được sử dụng.
            - color: Màu của văn bản.
            - x: Tọa độ x của văn bản.
            - y: Tọa độ y của văn bản.
            - font_size: Kích thước của font chữ.
        """
        font = pygame.font.Font(None, font_size)
        text_obj = font.render(text, True, color)
        text_rect = text_obj.get_rect()
        text_rect.center = (x, y)
        self.screen.blit(text_obj, text_rect)

    
    def apply_action(self):
        """
        Get the event from the event queue and run the appropriate 
        action.
        """
        global a, b
        # Take the event from the event queue.
        for ev in pygame.event.get():
            
            # Check if the close button was fired.
            con_tro = list(main_app.con_tro)
            x, y = con_tro[0], con_tro[1]
            if (267 <= x <= 335 ) and (0 <= y <= 10):
                self.done = True
            # Detect the key evevents for game control.
            gesture = main_app.gesture
            # print(gesture, 2)
            a = gesture
            print(a, b)
            # if (gesture is not None) and (a != b):
            if (gesture is not None):
                b = a
                pygame.time.delay(200)
                
                if gesture == 'Down':
                    self.active_block.move(0,constants.BHEIGHT)
                if gesture == 'Left':
                    self.active_block.move(-constants.BWIDTH,0)
                if gesture == 'Right':
                    self.active_block.move(constants.BWIDTH,0)
                if gesture == 'Rotate':
                    self.active_block.rotate()
                if gesture == 'Nothing':
                    self.draw_pause_menu("Game pause", self.active_block.color)
                if gesture == 'Click':
                    pass
                
            
            # Detect if the movement event was fired by the timer.
            if ev.type == constants.TIMER_MOVE_EVENT:
                self.active_block.move(0,constants.BHEIGHT)

    def set_move_timer(self):
        """
        Setup the move timer to the 
        """
        # Setup the time to fire the move event. Minimal allowed value is 1
        speed = math.floor(constants.MOVE_TICK / self.speed)
        speed = max(1,speed)
        pygame.time.set_timer(constants.TIMER_MOVE_EVENT,speed)
    
    def draw_tutorial(self, i):
        self.screen.fill((0, 0, 0))
        image_path = constants.IMAGE_TUTORIAL[i]  # Đường dẫn đến hình ảnh của bạn
        image_load = pygame.image.load(image_path)
        image_scale = pygame.transform.scale(image_load , (250, 80))
        self.screen.blit(image_scale, (45, 30))
        if self.active_block != None:
            self.active_block.draw()
        else:
            self.draw_pause_menu("Next step", constants.CYAN)
        pygame.display.flip()
        
    def run_tutorial(self):
        
        pygame.init()
        pygame.font.init()
        self.myfont = pygame.font.SysFont(pygame.font.get_default_font(),constants.FONT_SIZE)
        self.screen = pygame.display.set_mode((self.resx,self.resy))
        pygame.display.set_caption("Tetris")    
  
        i = 0
        while i < 4:
            white = (255, 255, 255)
            BLACK = (0, 0, 0)

# Hình ảnh

            self.get_block_new(150, 250)
            
            self.draw_tutorial(i)
            while True:
                ok = 0
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.unicode == 'q'):
                        self.draw_pause_menu("Tutorial pause", constants.CYAN)
                    if ev.type == pygame.KEYDOWN:
                        
                        gesture = main_app.gesture
                        if gesture == constants.O_TUTORIAL[i]:
                            if (i == 3):
                                self.active_block.rotate()
                                ok = 1
                                
                                self.draw_tutorial(i)
                                pygame.time.delay(2000)
                                break
                            else:
                                self.active_block.move(constants.MOVE_TUTORIAL[i][0], constants.MOVE_TUTORIAL[i][1])
                                ok = 1
                                self.draw_tutorial(i)
                                
                                pygame.time.delay(2000)
                                break
                if ok:
                    break       
            self.active_block = None
            self.draw_tutorial(i)
            # continue_button_background = pygame.Rect(10, 10,100, 30)
            # pygame.draw.rect(self.screen, BLACK, continue_button_background, border_radius=10)
            # continue_button = pygame.Rect(top = 10, left = 10, width=100, height=30)
            # pygame.draw.rect(self.screen, white, continue_button, 2, 10)

            i += 1
        ScreenManager.Run_game(16, 30)
    def run(self):
        
        # Initialize the game (pygame, fonts)
        pygame.init()
        pygame.font.init()
        self.myfont = pygame.font.SysFont(pygame.font.get_default_font(),constants.FONT_SIZE)
        self.screen = pygame.display.set_mode((self.resx,self.resy))
        pygame.display.set_caption("Tetris")
        # Setup the time to fire the move event every given time
        self.set_move_timer()
        # Control variables for the game. The done signal is used 
        # to control the main loop (it is set by the quit action), the game_over signal
        # is set by the game logic and it is also used for the detection of "game over" drawing.
        # Finally the new_block variable is used for the requesting of new tetris block. 
        self.done = False
        self.game_over = False
        self.new_block = True
        # Print the initial score
        self.print_status_line()
        while not(self.done) and not(self.game_over):
            # Get the block and run the game logic
            self.get_block()
            self.game_logic()
            self.draw_game()
        # Display the game_over and wait for a keypress
        if self.game_over:
            self.draw_pause_menu("Game over!")
        # Disable the pygame stuff
        pygame.font.quit()
        pygame.display.quit()        
   
    def print_status_line(self):
        """
        Print the current state line
        """
        string = ["SCORE: {0}   SPEED: {1}x".format(self.score,self.speed)]
        self.print_text(string,constants.POINT_MARGIN,constants.POINT_MARGIN)        



    def print_text(self,str_lst,x,y):
        """
        Print the text on the X,Y coordinates. 

        Parameters:
            - str_lst - list of strings to print. Each string is printed on new line.
            - x - X coordinate of the first string
            - y - Y coordinate of the first string
        """
        prev_y = 0
        for string in str_lst:
            size_x,size_y = self.myfont.size(string)
            txt_surf = self.myfont.render(string,False,(255,255,255))
            self.screen.blit(txt_surf,(x,y+prev_y))
            prev_y += size_y 

    def print_center(self,str_list):
        """
        Print the string in the center of the screen.
        
        Parameters:
            - str_lst - list of strings to print. Each string is printed on new line.
        """
        max_xsize = max([tmp[0] for tmp in map(self.myfont.size,str_list)])
        self.print_text(str_list,self.resx/2-max_xsize/2,self.resy/2)

    def block_colides(self):
        """
        Check if the block colides with any other block.

        The function returns True if the collision is detected.
        """
        for blk in self.blk_list:
            # Check if the block is not the same
            if blk == self.active_block:
                continue 
            # Detect situations
            if(blk.check_collision(self.active_block.shape)):
                return True
        return False

    def game_logic(self):
        """
        Implementation of the main game logic. This function detects colisions
        and insertion of new tetris blocks.
        """
        # Remember the current configuration and try to 
        # apply the action
        self.active_block.backup()
        self.apply_action()
        # Border logic, check if we colide with down border or any
        # other border. This check also includes the detection with other tetris blocks. 
        down_board  = self.active_block.check_collision([self.board_down])
        if down_board:
            self.active_block.restore()
            self.active_block.backup()
            self.active_block.move(0,constants.BHEIGHT)
        down_board  = self.active_block.check_collision([self.board_down])
            
        any_border  = self.active_block.check_collision([self.board_left,self.board_up,self.board_right])
        block_any   = self.block_colides()
        # Restore the configuration if any collision was detected
        if down_board or any_border or block_any:
            self.active_block.restore()
        # So far so good, sample the previous state and try to move down (to detect the colision with other block). 
        # After that, detect the the insertion of new block. The block new block is inserted if we reached the boarder
        # or we cannot move down.
        self.active_block.backup()
        self.active_block.move(0,constants.BHEIGHT)
        can_move_down = not self.block_colides()  
        self.active_block.restore()
        # We end the game if we are on the respawn and we cannot move --> bang!
        if not can_move_down and (self.start_x == self.active_block.x and self.start_y == self.active_block.y):
            self.game_over = True
        # The new block is inserted if we reached down board or we cannot move down.
        if down_board or not can_move_down:     
            # Request new block
            self.new_block = True
            # Detect the filled line and possibly remove the line from the 
            # screen.
            self.detect_line()   
 
    def detect_line(self):
        """
        Detect if the line is filled. If yes, remove the line and
        move with remaining bulding blocks to new positions.
        """
        # Get each shape block of the non-moving tetris block and try
        # to detect the filled line. The number of bulding blocks is passed to the class
        # in the init function.
        for shape_block in self.active_block.shape:
            tmp_y = shape_block.y
            tmp_cnt = self.get_blocks_in_line(tmp_y)
            # Detect if the line contains the given number of blocks
            if tmp_cnt != self.blocks_in_line:
                continue 
            # Ok, the full line is detected!     
            self.remove_line(tmp_y)
            # Update the score.
            self.score += self.blocks_in_line * constants.POINT_VALUE 
            # Check if we need to speed up the game. If yes, change control variables
            if self.score > self.score_level:
                self.score_level *= constants.SCORE_LEVEL_RATIO
                self.speed       *= constants.GAME_SPEEDUP_RATIO
                # Change the game speed
                self.set_move_timer()

    def remove_line(self,y):
        """
        Remove the line with given Y coordinates. Blocks below the filled
        line are untouched. The rest of blocks (yi > y) are moved one level done.

        Parameters:
            - y - Y coordinate to remove.
        """ 
        # Iterate over all blocks in the list and remove blocks with the Y coordinate.
        for block in self.blk_list:
            block.remove_blocks(y)
        # Setup new block list (not needed blocks are removed)
        self.blk_list = [blk for blk in self.blk_list if blk.has_blocks()]

    def get_blocks_in_line(self,y):
        """
        Get the number of shape blocks on the Y coordinate.

        Parameters:
            - y - Y coordinate to scan.
        """
        # Iteraveovel all block's shape list and increment the counter
        # if the shape block equals to the Y coordinate.
        tmp_cnt = 0
        for block in self.blk_list:
            for shape_block in block.shape:
                tmp_cnt += (1 if y == shape_block.y else 0)            
        return tmp_cnt

    def draw_board(self):
        """
        Draw the white board.
        """
        pygame.draw.rect(self.screen,constants.WHITE,self.board_up)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_down)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_left)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_right)
        # Update the score         
        self.print_status_line()

    def get_block(self):
        """
        Generate new block into the game if is required.
        """
        if self.new_block:
            # Get the block and add it into the block list(static for now)
            tmp = random.randint(0,len(self.block_data)-1)
            data = self.block_data[tmp]
            self.active_block = block.Block(data[0],self.start_x,self.start_y,self.screen,data[1],data[2])
            self.blk_list.append(self.active_block)
            self.new_block = False
    def get_block_new(self, start_x, start_y):
        """
        Generate new block into the game if is required.
        """
        
            # Get the block and add it into the block list(static for now)
        tmp = random.randint(0,len(self.block_data)-1)
        data = self.block_data[tmp]

        self.active_block = block.Block(data[0],start_x,start_y,self.screen,data[1],data[2])

    def draw_game(self):
        """
        Draw the game screen.
        """
        # Clean the screen, draw the board and draw
        # all tetris blocks
        
        self.screen.fill(constants.BLACK)
        self.draw_board()
        for blk in self.blk_list:
            blk.draw()
        # Draw the screen buffer
        pygame.display.flip()
    def draw_rounded_rect(surface, rect, color, corner_radius):
    # Tạo hình chữ nhật bo góc
        rect_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(rect_surf, color, rect_surf.get_rect(), border_radius=corner_radius)
        surface.blit(rect_surf, rect.topleft)
    def draw_pause_menu(self, text, color):
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        font = pygame.font.Font(None, 36)

        while True:
           

            # Vẽ thanh pause
            pause_color = color
            
            # Tạo đối tượng Rect với bo tròn góc
            pause_rect = pygame.Rect(self.resx // 8, self.resy // 3, self.resx * 6 // 8, self.resy // 9)
           
            pygame.draw.rect(self.screen, pause_color, pause_rect, width=5,border_radius=15)
            self.draw_text(text, font, WHITE, pause_rect.centerx, pause_rect.y + 25, 40)

            # Vẽ các lựa chọn
            continue_button_background = pygame.Rect(pause_rect.left + 20, pause_rect.bottom - 20, 100, 30)
            pygame.draw.rect(self.screen, BLACK, continue_button_background, border_radius=10)
            continue_button = pygame.Rect(pause_rect.left + 20, pause_rect.bottom - 20, 100, 30)
            pygame.draw.rect(self.screen, pause_color, continue_button, 2, 10)

            self.draw_text('Continue', font, WHITE, continue_button.centerx, continue_button.y + 15, 30)
            
            quit_button_background = pygame.Rect(pause_rect.right - 120, pause_rect.bottom - 20, 100, 30)
            pygame.draw.rect(self.screen, BLACK, quit_button_background, border_radius=10)
            quit_button = pygame.Rect(pause_rect.right - 120, pause_rect.bottom - 20, 100, 30)
            self.draw_text('Quit', font, WHITE, quit_button.centerx, quit_button.y + 15, 30)
            pygame.draw.rect(self.screen, pause_color, quit_button, 2, 10)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos

                    # Kiểm tra xem chuột ở đâu
                    if continue_button.collidepoint(mouse_pos):
                        if text == "Game over!":
                            Tetris.run(self.bx, self.by)
                        return 
                    elif quit_button.collidepoint(mouse_pos):
                        # Thực hiện hành động khi nhấp vào nút "Quit"
                        return ScreenManager.Run_game(self.bx, self.by)

            pygame.display.update()
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
        background_image = pygame.image.load(".\\items\\background_menu.png")


        background_image = pygame.transform.scale(background_image, (self.screen_width, self.screen_height))
        menu_image = pygame.image.load(".\\items\\start_game.png")
        menu_image = pygame.transform.scale(menu_image, (230, 230))
        while True:
            
            self.screen.blit(background_image, (0, 0))
        
            self.screen.blit(menu_image, (self.screen_width // 7, 10))
            # Vẽ các lựa chọn
            play_button = pygame.Rect(self.screen_width // 2 - 100, 200, 200, 50)
            
            pygame.draw.rect(self.screen, WHITE, play_button, border_radius=10)
            self.draw_text('Play', font, BLACK, self.screen_width // 2, 225)
            instructions_button = pygame.Rect(self.screen_width // 2 - 100, 300, 200, 50)
            pygame.draw.rect(self.screen, WHITE, instructions_button, border_radius=10)
            self.draw_text('Tutorial', font, BLACK, self.screen_width // 2, 325)
            

            quit_button = pygame.Rect(self.screen_width // 2 - 100, 400, 200, 50)
            pygame.draw.rect(self.screen, WHITE, quit_button, border_radius=10)
            self.draw_text('Quit', font, BLACK, self.screen_width // 2, 425)
            

            while True:

                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEMOTION:
                        mouse_pos = event.pos
                        if play_button.collidepoint(mouse_pos):
                            pygame.draw.rect(self.screen, BLACK, play_button, border_radius=10)
                            self.draw_text('Play', font, WHITE, self.screen_width // 2, 225)
                            pygame.display.update()
                        # Thực hiện hành động khi chuột di chuyển vào ô chơi
                        elif instructions_button.collidepoint(mouse_pos):
                            pygame.draw.rect(self.screen, BLACK, instructions_button, border_radius=10)
                            self.draw_text('Tutorial', font, WHITE, self.screen_width // 2, 325)
                            pygame.display.update()
                        # Thực hiện hành động khi chuột di chuyển vào ô hướng dẫn
                        elif quit_button.collidepoint(mouse_pos):
                            pygame.draw.rect(self.screen, BLACK, quit_button, border_radius=10)
                            self.draw_text('Quit', font, WHITE, self.screen_width // 2, 425)
                            pygame.display.update()
                        else:
                            pygame.draw.rect(self.screen, WHITE, play_button, border_radius=10)
                            self.draw_text('Play', font, BLACK, self.screen_width // 2, 225)
                            pygame.draw.rect(self.screen, WHITE, instructions_button, border_radius=10)
                            self.draw_text('Tutorial', font, BLACK, self.screen_width // 2, 325)
                            pygame.draw.rect(self.screen, WHITE, quit_button, border_radius=10)
                            self.draw_text('Quit', font, BLACK, self.screen_width // 2, 425)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = event.pos

                        # Kiểm tra xem chuột ở đâu
                        if play_button.collidepoint(mouse_pos):
                            # Thực hiện hành động khi nhấp vào nút "Chơi"
                            # Đây là nơi bạn gọi hàm để chuyển sang màn hình chơi game
                            
                            Tetris(self.x, self.y).run()

                            
                        elif instructions_button.collidepoint(mouse_pos):
                            # Thực hiện hành động khi nhấp vào nút "Hướng dẫn"
                            Tetris(self.x, self.y).run_tutorial()
                        elif quit_button.collidepoint(mouse_pos):
                            # Thực hiện hành động khi nhấp vào nút "Thoát"
                            sys.exit()

                pygame.display.update()

# Sử dụng
    def Run_game(x, y):
        pygame.init()
        screen_width = x*constants.BWIDTH+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        screen_height = y*constants.BHEIGHT+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        print(screen_height, screen_width)
        screen_manager = ScreenManager(screen_width, screen_height, x, y)
        return screen_manager.draw_menu()
def Run(x, y):

    ScreenManager.Run_game(x, y)


#Special add to try pull requests
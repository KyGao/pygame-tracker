# KidsCanCode - Game Development with Pygame video series
# Jumpy! (a platform game) - Part 18
# Video link: https://youtu.be/i0PaigPo6KM
# scrolling cloud background
# Art from Kenney.nl
# Happy Tune by http://opengameart.org/users/syncopika
# Yippee by http://opengameart.org/users/snabisch


import pygame as pg
import random
from settings import *
from sprites import *
from os import path
import cv2



################################################
import numpy as np
import cv2
import sys
from time import time

import kcftracker






            
#################################################


class Game:
    def __init__(self):
        # initialize game window, etc
        pg.init()
        pg.mixer.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.running = True
        self.font_name = pg.font.match_font(FONT_NAME)
        self.load_data()

    def load_data(self):
        # load high score
        self.dir = path.dirname(__file__)
        with open(path.join(self.dir, HS_FILE), 'r') as f:
            try:
                self.highscore = int(f.read())
            except:
                self.highscore = 0
        # load spritesheet image
        img_dir = path.join(self.dir, 'img')
        self.spritesheet = Spritesheet(path.join(img_dir, SPRITESHEET))
        # cloud images
        self.cloud_images = []
        for i in range(1, 4):
            self.cloud_images.append(pg.image.load(path.join(img_dir, 'cloud{}.png'.format(i))).convert())
        # load sounds
        self.snd_dir = path.join(self.dir, 'snd')
        self.jump_sound = pg.mixer.Sound(path.join(self.snd_dir, 'Jump33.wav'))
        self.boost_sound = pg.mixer.Sound(path.join(self.snd_dir, 'Boost16.wav'))

    def new(self):
        # start a new game
        self.score = 0
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.platforms = pg.sprite.Group()
        self.powerups = pg.sprite.Group()
        self.mobs = pg.sprite.Group()
        self.clouds = pg.sprite.Group()
        self.player = Player(self)
        for plat in PLATFORM_LIST:
            Platform(self, *plat)
        self.mob_timer = 0
        pg.mixer.music.load(path.join(self.snd_dir, 'Happy Tune.ogg'))
        for i in range(8):
            c = Cloud(self)
            c.rect.y += 500
        self.run()

    def run(self):
        # Game Loop
        '''
        pg.mixer.music.play(loops=-1)
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()
        pg.mixer.music.fadeout(500)
        '''
    ##########################################
    
        pg.mixer.music.play(loops=-1)
        self.playing = True
        
        self.selectingObject = False
        self.initTracking = False
        self.onTracking = False
        self.ix, self.iy, self.cx, self.cy = -1, -1, -1, -1
        self.w, self.h = 0, 0

        duration = 0.01

        last_y = 0
        
        def draw_boundingbox(event, x, y, flags, param):
            global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

            if event == cv2.EVENT_LBUTTONDOWN:
                self.selectingObject = True
                self.onTracking = False
                self.ix, self.iy = x, y
                self.cx, self.cy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                self.cx, self.cy = x, y

            elif event == cv2.EVENT_LBUTTONUP:
                self.selectingObject = False
                if (abs(x - self.ix) > 10 and abs(y - self.iy) > 10):
                    self.w, self.h = abs(x - self.ix), abs(y - self.iy)
                    self.ix, self.iy = min(x, self.ix), min(y, self.iy)
                    self.initTracking = True
                else:
                    self.onTracking = False

            elif event == cv2.EVENT_RBUTTONDOWN:
                self.onTracking = False
                if (self.w > 0):
                    self.ix, self.iy = x - self.w / 2, y - self.h / 2
                    self.initTracking = True
        
        cap = cv2.VideoCapture(0)
        tracker = kcftracker.KCFTracker(True, True, True)
        cv2.namedWindow('tracking')
        cv2.setMouseCallback('tracking',draw_boundingbox)
        jump_cnt = 0  # count jump time
        while cap.isOpened() and self.playing:
            ret, frame = cap.read()
            if not ret:
                break

            if (self.selectingObject):
                cv2.rectangle(frame, (self.ix, self.iy), (self.cx, self.cy), (0, 255, 255), 1)
            elif (self.initTracking):
                cv2.rectangle(frame, (self.ix, self.iy), (self.ix + self.w, self.iy + self.h), (0, 255, 255), 2)

                tracker.init([self.ix, self.iy, self.w, self.h], frame)
                last_y = self.iy

                self.initTracking = False
                self.onTracking = True
            elif (self.onTracking):

                t0 = time()
                boundingbox = tracker.update(frame)
                t1 = time()

                boundingbox = list(map(int, boundingbox))
                center = [boundingbox[0] + boundingbox[2] / 2, boundingbox[1] + boundingbox[3] / 2]
                cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),(boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)
                duration = 0.8 * duration + 0.2 * (t1 - t0)
                # duration = t1-t0
                
                cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
                
                if jump_cnt <= 0:
                    jump_cnt = last_y - boundingbox[1]
                self.clock.tick(FPS)
                self.events(jump_cnt > 0)
                self.update(center)
                self.draw()
                jump_cnt -= 10
                
            cv2.imshow('tracking', frame)
            c = cv2.waitKey(inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break

        pg.mixer.music.fadeout(500)

    ################################################

    def update(self, center):
        # Game Loop - Update
        self.all_sprites.update(center)

        # spawn a mob?
        now = pg.time.get_ticks()
        if now - self.mob_timer > 5000 + random.choice([-1000, -500, 0, 500, 1000]):
            self.mob_timer = now
            Mob(self)
        # hit mobs?
        mob_hits = pg.sprite.spritecollide(self.player, self.mobs, False, pg.sprite.collide_mask)
        if mob_hits:
            self.playing = False

        # check if player hits a platform - only if falling
        if self.player.vel.y > 0:
            hits = pg.sprite.spritecollide(self.player, self.platforms, False)
            if hits:
                lowest = hits[0]
                for hit in hits:
                    if hit.rect.bottom > lowest.rect.bottom:
                        lowest = hit
                if self.player.pos.x < lowest.rect.right + 10 and \
                   self.player.pos.x > lowest.rect.left - 10:
                    if self.player.pos.y < lowest.rect.centery:
                        self.player.pos.y = lowest.rect.top
                        self.player.vel.y = 0
                        self.player.jumping = False

        # if player reaches top 1/4 of screen
        if self.player.rect.top <= HEIGHT / 4:
            if random.randrange(100) < 15:
                Cloud(self)
            self.player.pos.y += max(abs(self.player.vel.y), 2)
            for cloud in self.clouds:
                cloud.rect.y += max(abs(self.player.vel.y / 2), 2)
            for mob in self.mobs:
                mob.rect.y += max(abs(self.player.vel.y), 2)
            for plat in self.platforms:
                plat.rect.y += max(abs(self.player.vel.y), 2)
                if plat.rect.top >= HEIGHT:
                    plat.kill()
                    self.score += 10

        # if player hits powerup
        pow_hits = pg.sprite.spritecollide(self.player, self.powerups, True)
        for pow in pow_hits:
            if pow.type == 'boost':
                self.boost_sound.play()
                self.player.vel.y = -BOOST_POWER
                self.player.jumping = False

        # Die!
        if self.player.rect.bottom > HEIGHT:
            for sprite in self.all_sprites:
                sprite.rect.y -= max(self.player.vel.y, 10)
                if sprite.rect.bottom < 0:
                    sprite.kill()
        if len(self.platforms) == 0:
            self.playing = False

        # spawn new platforms to keep same average number
        while len(self.platforms) < 6:
            width = random.randrange(50, 100)
            Platform(self, random.randrange(0, WIDTH - width),
                     random.randrange(-75, -30))

    def events(self, jump):
        # Game Loop - events
        for event in pg.event.get():
            # check for closing window
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False
        if jump:
            self.player.jump()
        else:
            self.player.jump_cut()
                
            '''
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    self.player.jump()
            if event.type == pg.KEYUP:
                if event.key == pg.K_SPACE:
                    self.player.jump_cut()
            '''
    def draw(self):
        # Game Loop - draw
        self.screen.fill(BGCOLOR)
        self.all_sprites.draw(self.screen)
        self.draw_text(str(self.score), 22, WHITE, WIDTH / 2, 15)
        # *after* drawing everything, flip the display
        pg.display.flip()

    def show_start_screen(self):
        # game splash/start screen
        pg.mixer.music.load(path.join(self.snd_dir, 'Yippee.ogg'))
        pg.mixer.music.play(loops=-1)
        self.screen.fill(BGCOLOR)
        self.draw_text("Guidance", 48, WHITE, WIDTH / 2, HEIGHT / 4)
        self.draw_text("Use mouse to draw a rectangle, ",22, WHITE, WIDTH/2, HEIGHT/4+90)
        self.draw_text("selecting the moving object", 22, WHITE, WIDTH / 2, HEIGHT / 4+120)
        self.draw_text("Move up to jump!", 22, WHITE, WIDTH / 2, HEIGHT / 2)
        self.draw_text("Press a key to continue", 22, WHITE, WIDTH / 2, HEIGHT * 3 / 4)
        self.draw_text("High Score: " + str(self.highscore), 22, WHITE, WIDTH / 2, 15)
        pg.display.flip()
        self.wait_for_key()
        pg.mixer.music.fadeout(500)

    def show_go_screen(self):
        # game over/continue
        if not self.running:
            return
        pg.mixer.music.load(path.join(self.snd_dir, 'Yippee.ogg'))
        pg.mixer.music.play(loops=-1)
        self.screen.fill(BGCOLOR)
        self.draw_text("GAME OVER", 48, WHITE, WIDTH / 2, HEIGHT / 4)
        self.draw_text("Score: " + str(self.score), 22, WHITE, WIDTH / 2, HEIGHT / 2)
        self.draw_text("Press a key to play again", 22, WHITE, WIDTH / 2, HEIGHT * 3 / 4)
        if self.score > self.highscore:
            self.highscore = self.score
            self.draw_text("NEW HIGH SCORE!", 22, WHITE, WIDTH / 2, HEIGHT / 2 + 40)
            with open(path.join(self.dir, HS_FILE), 'w') as f:
                f.write(str(self.score))
        else:
            self.draw_text("High Score: " + str(self.highscore), 22, WHITE, WIDTH / 2, HEIGHT / 2 + 40)
        pg.display.flip()
        self.wait_for_key()
        pg.mixer.music.fadeout(500)

    def wait_for_key(self):
        waiting = True
        while waiting:
            self.clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    waiting = False
                    self.running = False
                if event.type == pg.KEYUP:
                    waiting = False

    def draw_text(self, text, size, color, x, y):
        font = pg.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(text_surface, text_rect)


'''
g = Game()
g.show_start_screen()
while g.running:
    g.new()
    g.show_go_screen()

pg.quit()
'''

#################################

g = Game()
g.show_start_screen()
while g.running:
    g.new()
    g.show_go_screen()



cap.release()
cv2.destroyAllWindows()
    
print("Bye")
pygame.quit()

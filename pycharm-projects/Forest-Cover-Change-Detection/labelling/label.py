

"""
    source code taken from https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558
    Edited by: Annus Zulfiqar
"""


from __future__ import print_function, division
from PIL import Image
import pygame
import sys
pygame.init()


class controls(object):
    keyboard = {
        'up': pygame.K_UP,
        'down': pygame.K_DOWN,
        'right': pygame.K_RIGHT,
        'left': pygame.K_LEFT
    }
    mouse = {
        'left_mouse_button': 1,
        'wheel_button': 2,
        'right_mouse_button': 3,
        'mouse_wheel_up': 4,
        'mouse_wheel_down': 5
    }
    reverse_keyboard = {val:key for key,val in keyboard.items()}
    reverse_mouse = {val:key for key,val in mouse.items()}
    pass

W, H = 1000, 700


def displayImage(screen, px, topleft, prior, refresh=False, rect=None):
    if not refresh:
        # ensure that the rect always has positive width, height
        x, y = topleft
        width = pygame.mouse.get_pos()[0] - topleft[0]
        height = pygame.mouse.get_pos()[1] - topleft[1]
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)

        # eliminate redundant drawing cycles (when mouse isn't moving)
        current = x, y, width, height
        if not (width and height):
            return current
        if current == prior:
            return current

        # draw transparent box and blit it onto canvas
        if not rect:
            screen.blit(px, px.get_rect())
        screen.blit(px, rect)
        im = pygame.Surface((width, height))
        im.fill((128, 128, 128))
        if prior is not None:
            # if not rect:
            pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
            # pygame.draw.rect(im, (32, 32, 32), rect, 1)
        im.set_alpha(128)
        screen.blit(im, (x, y))
        pygame.display.flip()

        # return current box extents
        return (x, y, width, height)

    else:
        # screen.blit(px, px.get_rect())
        pass


def setup(image_path, screen_size, screen, px, rect=None, first=False):
    if first:
        px = pygame.image.load(image_path)
        screen = pygame.display.set_mode(screen_size)
    if rect:
        screen.blit(px, rect)
    else:
        screen.blit(px, px.get_rect())
    pygame.display.update()
    return screen, px


def translate_surface(px, screen, x_shift, y_shift):
    screen.blit(px, (x_shift, y_shift))
    # pygame.display.flip()
    return screen, px


def mainLoop(screen, px, input_loc, screen_size):
    topleft = bottomright = prior = None
    current_position = px.get_rect()
    reset = False
    while bottomright is None:
        for event in pygame.event.get():
            # this is just for quitting the labeling tool
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                return [None] * 4  # ???
            if event.type == pygame.KEYDOWN:
                key = event.key
                if key in controls.reverse_keyboard.keys():
                    log = "\rlog: {}".format(controls.reverse_keyboard[key])
                    print(log, end='')
                    sys.stdout.flush()
                # exit?
                if key == pygame.K_ESCAPE or key == pygame.K_q:
                    pygame.display.quit()
                    pygame.quit()
                    return [None]*4 #???
                # if we want to translate the image on the window
                if key in controls.keyboard.values():
                    if key == controls.keyboard['right']:
                        current_position.x += 10
                        reset = True
                    if key == controls.keyboard['left']:
                        current_position.x -= 10
                        reset = True
                    if key == controls.keyboard['up']:
                        current_position.y -= 10
                        reset = True
                    if key == controls.keyboard['down']:
                        current_position.y += 10
                        reset = True

            if event.type == pygame.MOUSEBUTTONDOWN:
                # all mouse buttons, even wheel button and turn
                button = event.button
                if button in controls.reverse_mouse.keys():
                    log = "\rlog: {}".format(controls.reverse_mouse[button])
                    print(log, end='')
                    sys.stdout.flush()
                if button == controls.mouse['right_mouse_button']:
                    if not topleft:
                        continue
                    else:
                        reset = True
                        topleft = None
                        prior = None
                if button == controls.mouse['left_mouse_button']:
                    if topleft == None:
                        pos = event.pos
                        topleft = (pos[0]-current_position.x, pos[1]-current_position[1])
                    else:
                        pos = event.pos
                        bottomright = (pos[0]-current_position.x, pos[1]-current_position[1])
        if topleft:
            prior = displayImage(screen, px, topleft, prior, rect=current_position)
        if reset:
            setup(input_loc, screen_size=screen_size, screen=screen, px=px, rect=current_position)
            reset = False

    return (topleft + bottomright)


def main(input_loc):
    # input_loc = '/home/annus/Pictures/lena.jpeg'
    output_loc = 'out.png'
    screen_size = (W, H)
    screen, px = setup(input_loc, screen_size=screen_size, screen=None, px=None, first=True)
    left, upper, right, lower = mainLoop(screen, px, input_loc, screen_size)
    if left is None or upper is None or right is None or lower is None:
        return
    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    im = Image.open(input_loc)
    im = im.crop((left, upper, right, lower))
    pygame.display.quit()
    im.save(output_loc)
    pass


if __name__ == '__main__':
    main(input_loc='/home/annus/Pictures/science-wallpaper-high-quality-For-Desktop-Wallpaper.jpg')






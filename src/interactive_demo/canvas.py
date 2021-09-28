# -*- coding: utf-8 -*-
""" Adopted from https://github.com/foobar167/junkyard/blob/master/manual_image_annotation1/polygon/gui_canvas.py """
import os
import sys
import time
import math
import tkinter as tk

from tkinter import ttk
from PIL import Image, ImageTk


def handle_exception(exit_code=0):
    """ Use: @land.logger.handle_exception(0)
        before every function which could cast an exception """

    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                if exit_code != 0:  # if zero, don't exit from the program
                    sys.exit(exit_code)  # exit from the program

        return inner

    return wrapper


class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    @handle_exception(1)
    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    @handle_exception(1)
    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)


class CanvasImage:
    """ Display and zoom image """

    def __init__(self, canvas_frame, canvas):
        """ Initialize the ImageFrame """
        self.current_scale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.2  # zoom magnitude
        self.__previous_state = 0  # previous state of the keyboard
        # Create ImageFrame in placeholder widget
        self.__imframe = canvas_frame
        # Vertical and horizontal scrollbars for canvas
        self.hbar = AutoScrollbar(canvas_frame, orient='horizontal')
        self.vbar = AutoScrollbar(canvas_frame, orient='vertical')
        self.hbar.grid(row=1, column=0, sticky='we')
        self.vbar.grid(row=0, column=1, sticky='ns')
        # Add scroll bars to canvas
        self.canvas = canvas
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        self.vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__size_changed())  # canvas is resized
        self.canvas.bind('<Button-1>', self.__left_mouse_button)  # remember canvas position
        self.canvas.bind('<ButtonPress-3>', self.__right_mouse_button_pressed)  # remember canvas position
        self.canvas.bind('<ButtonPress-2>', self.__right_mouse_button_pressed)  # remember canvas position (MacOS)
        self.canvas.bind('<ButtonRelease-3>', self.__right_mouse_button_released)  # remember canvas position
        self.canvas.bind('<ButtonRelease-2>', self.__right_mouse_button_released)  # remember canvas position (MacOS)
        self.canvas.bind('<B3-Motion>', self.__right_mouse_button_motion)  # move canvas to the new position
        self.canvas.bind('<B2-Motion>', self.__right_mouse_button_motion)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        self.container = None

        self._click_callback = None

    def register_click_callback(self,  click_callback):
        self._click_callback = click_callback

    def reload_image(self, image, reset_canvas=True):
        self.__original_image = image.copy()
        self.__current_image = image.copy()

        if reset_canvas:
            self.imwidth, self.imheight = self.__original_image.size
            self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side

            scale = min(self.canvas.winfo_width() / self.imwidth, self.canvas.winfo_height() / self.imheight)
            if self.container:
                self.canvas.delete(self.container)

            self.container = self.canvas.create_rectangle((0, 0, scale * self.imwidth, scale * self.imheight), width=0)
            self.current_scale = scale
            self._reset_canvas_offset()

        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def __show_image(self):
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            border_width = 2
            sx1, sx2 = x1 / self.current_scale, x2 / self.current_scale
            sy1, sy2 = y1 / self.current_scale, y2 / self.current_scale
            crop_x, crop_y = max(0, math.floor(sx1 - border_width)), max(0, math.floor(sy1 - border_width))
            crop_w, crop_h = math.ceil(sx2 - sx1 + 2 * border_width), math.ceil(sy2 - sy1 + 2 * border_width)
            crop_w = min(crop_w, self.__original_image.width - crop_x)
            crop_h = min(crop_h, self.__original_image.height - crop_y)

            __current_image = self.__original_image.crop((crop_x, crop_y,
                                                          crop_x + crop_w, crop_y + crop_h))
            crop_zw = int(round(crop_w * self.current_scale))
            crop_zh = int(round(crop_h * self.current_scale))
            zoom_sx, zoom_sy = crop_zw / crop_w, crop_zh / crop_h
            crop_zx, crop_zy = crop_x * zoom_sx, crop_y * zoom_sy
            self.real_scale = (zoom_sx, zoom_sy)

            interpolation = Image.NEAREST if self.current_scale > 2.0 else Image.ANTIALIAS
            __current_image = __current_image.resize((crop_zw, crop_zh), interpolation)
            zx1, zy1 = x1 - crop_zx, y1 - crop_zy
            zx2 = min(zx1 + self.canvas.winfo_width(), __current_image.width)
            zy2 = min(zy1 + self.canvas.winfo_height(), __current_image.height)

            self.__current_image = __current_image.crop((zx1, zy1, zx2, zy2))

            imagetk = ImageTk.PhotoImage(self.__current_image)
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def _get_click_coordinates(self, event):
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)

        if self.outside(x, y):
            return None

        box_image = self.canvas.coords(self.container)
        x = max(x - box_image[0], 0)
        y = max(y - box_image[1], 0)

        x = int(x / self.real_scale[0])
        y = int(y / self.real_scale[1])

        return x, y

    # ================================================ Canvas Routines =================================================
    def _reset_canvas_offset(self):
        self.canvas.configure(scrollregion=(0, 0, 5000, 5000))
        self.canvas.scan_mark(0, 0)
        self.canvas.scan_dragto(int(self.canvas.canvasx(0)), int(self.canvas.canvasy(0)), gain=1)

    def _change_canvas_scale(self, relative_scale, x=0, y=0):
        new_scale = self.current_scale * relative_scale

        if new_scale > 20:
            return

        if new_scale * self.__original_image.width < self.canvas.winfo_width() and \
           new_scale * self.__original_image.height < self.canvas.winfo_height():
            return

        self.current_scale = new_scale
        self.canvas.scale('all', x, y, relative_scale, relative_scale)  # rescale all objects

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __size_changed(self):
        new_scale_w = self.canvas.winfo_width() / (self.current_scale * self.__original_image.width)
        new_scale_h = self.canvas.winfo_height() / (self.current_scale * self.__original_image.height)
        new_scale = min(new_scale_w, new_scale_h)
        if new_scale > 1.0:
            self._change_canvas_scale(new_scale)
        self.__show_image()

    # ================================================ Mouse callbacks =================================================  
    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area

        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120 or event.delta == 1:  # scroll down, zoom out, smaller
            scale /= self.__delta
        if event.num == 4 or event.delta == 120 or event.delta == -1:  # scroll up, zoom in, bigger
            scale *= self.__delta

        self._change_canvas_scale(scale, x, y)
        self.__show_image()
    
    def __left_mouse_button(self, event):
        if self._click_callback is None:
            return

        coords = self._get_click_coordinates(event)

        if coords is not None:
            self._click_callback(is_positive=True, x=coords[0], y=coords[1])

    def __right_mouse_button_pressed(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self._last_rb_click_time = time.time()
        self._last_rb_click_event = event
        self.canvas.scan_mark(event.x, event.y)

    def __right_mouse_button_released(self, event):
        time_delta = time.time() - self._last_rb_click_time
        move_delta = math.sqrt((event.x - self._last_rb_click_event.x) ** 2 +
                               (event.y - self._last_rb_click_event.y) ** 2)
        if time_delta > 0.5 or move_delta > 3:
            return

        if self._click_callback is None:
            return

        coords = self._get_click_coordinates(self._last_rb_click_event)

        if coords is not None:
            self._click_callback(is_positive=False, x=coords[0], y=coords[1])

    def __right_mouse_button_motion(self, event):
        """ Drag (move) canvas to the new position """
        move_delta = math.sqrt((event.x - self._last_rb_click_event.x) ** 2 +
                               (event.y - self._last_rb_click_event.y) ** 2)
        if move_delta > 3:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    # ================================================= Keys Callback ==================================================
    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            self.keycodes = {}  # init key codes
            if os.name == 'nt':  # Windows OS
                self.keycodes = {
                    'd': [68, 39, 102],
                    'a': [65, 37, 100],
                    'w': [87, 38, 104],
                    's': [83, 40, 98],
                }
            else:  # Linux OS
                self.keycodes = {
                    'd': [40, 114, 85],
                    'a': [38, 113, 83],
                    'w': [25, 111, 80],
                    's': [39, 116, 88],
                }
            if event.keycode in self.keycodes['d']:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in self.keycodes['a']:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in self.keycodes['w']:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in self.keycodes['s']:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll', 1, 'unit', event=event)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracking algorithm for detecting circular particles in a video sequence.

Modified version:
- Added ROI selection for first-frame detection using mouse drag
- Fixed shape/indexing issues and redundant code
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, time, pickle, os
from matplotlib.widgets import RectangleSelector


class data(object):
    """Simple container class for storing data"""
    pass


class Tracking(object):
    def __init__(self, display=True, filepath=''):
        self.data = data()
        self.timestamps = []
        self.filepath = ''.join(filepath)
        self.folderpath = ''.join(filepath[:2])
        self.filename = filepath[2]
        self.picklepath = os.path.join(self.folderpath, 'pickle')
        self.input = True
        self.overwrite = False
        self.roi_coords = None  # for user-selected ROI

        if display:
            print(f'File: {self.filepath}')

    def set_parameters(self, params):
        self.params = params
        self.p2ini = params['p2']
        self.overwrite = params.get('overwrite', False)

    def start_tracking(self):
        print("### START TRACKING ###")
        self.f = self.params['t0']
        self.cap = cv2.VideoCapture(self.filepath)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f)
        
        self.totalframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.totalframes <= 0:
            self.totalframes = 1

        self.frames_to_track = int(min(self.params['tf'] - self.params['t0'], self.totalframes))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return

        print(f'TOTAL FRAMES: {self.totalframes}, t0 = {self.params["t0"]}')
        self.read_frame()
        print(f'IMAGE RESOLUTION: {self.cimg.shape}')
        self.findcircles()
        self.N = len(self.circles)
        print(f'Number of circles detected: {self.N}')

        self.data.raw = np.zeros((self.frames_to_track, self.N, 3))
        self.data.raw[0, :, :] = np.asarray(self.circles)
        print(f'circle radius ≈ {np.mean(self.circles[:,2]):.2f}px')

        self.f += 1
        print(f'### ITERATE ### on {self.N} particles')
        self.run_tracker()

    def run_tracker(self):
        self.f = self.params['t0'] + 1
        while self.f < self.totalframes and self.f < self.params['tf']:
            print(self.f)
            self.read_frame()
            if not self.ret:
                print('End of tracking')
                break

            self.findcircles()
            try:
                self.data.raw[self.f - self.params['t0'], :, :] = np.asarray(self.circles)
            except ValueError:
                print('Particle number not conserved')
                break
            self.f += 1

        self.savedata()

    # ---------------------------------------------------------
    # ROI Selection for first frame
    # ---------------------------------------------------------
    
    def select_ROI(self):
        """User drags a box to define ROI for initial detection."""
        self.roi_coords = []

        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            self.roi_coords = [int(min(x1, x2)), int(max(x1, x2)),
                            int(min(y1, y2)), int(max(y1, y2))]
            print(f"ROI selected: x=[{self.roi_coords[0]}, {self.roi_coords[1]}], "
                f"y=[{self.roi_coords[2]}, {self.roi_coords[3]}]")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cv2.cvtColor(self.cimg, cv2.COLOR_GRAY2BGR))
        ax.set_title("Drag to select ROI for initial detection,\nthen press ENTER or close window")

        toggle_selector = RectangleSelector(
            ax, line_select_callback,
            useblit=True,
            button=[1],  # left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        # ✅ Keep plot active until Enter or window closed
        print(">>> Draw a rectangle with the mouse, then press ENTER to confirm.")
        plt.connect('key_press_event', lambda event: plt.close(fig) if event.key in ['enter', 'return'] else None)
        plt.show()

        if not self.roi_coords:
            print("No ROI selected, using full frame.")
            h, w = self.cimg.shape
            self.roi_coords = [0, w, 0, h]
    # ---------------------------------------------------------

    def findcircles(self):
        if self.f == self.params['t0']:  # first frame
            if not self.roi_coords:
                self.select_ROI()

            x_min, x_max, y_min, y_max = self.roi_coords
            roi = self.cimg[y_min:y_max, x_min:x_max]

            circles = self.circlefitter(roi)
            if len(circles):
                circles = np.array(circles)
                circles[:, 0] += x_min
                circles[:, 1] += y_min
            self.circles = circles

            self.photocheck()
            time.sleep(0.3)
            plt.close()

        else:  # following frames
            coords = self.circles
            circles = []
            for n, xy in enumerate(coords):
                self.params['p2'] = self.p2ini
                y, x = int(xy[1]), int(xy[0])
                ROIr = self.params['r_ROI']

                x_min = max(x - ROIr, 0)
                x_max = min(x + ROIr, self.cimg.shape[1])
                y_min = max(y - ROIr, 0)
                y_max = min(y + ROIr, self.cimg.shape[0])

                self.ROIimg = self.cimg[y_min:y_max, x_min:x_max]
                i = 0
                while True:
                    circle = self.circlefitter(self.ROIimg)
                    
                    
                    if i > 7 or self.params['p2'] < 1:
                        print('Cannot find single particle, entering manual mode')
                        circle = self.manual_detection()
                        circle[0, :2] += [x_min, y_min]
                        circles.append(list(circle[0]))
                        break
                    if len(circle) == 0:
                        i += 1
                        self.decrease_threshold()
                    elif len(circle) > 1:
                        i += 1
                        self.increase_threshold()
                    else:
                        circle[0, :2] += [x_min, y_min]
                        circles.append(list(circle[0]))
                        break
            self.circles = np.asarray(circles)

    def savedata(self):
        self.data.timestamps = self.timestamps
        self.extract_data()

    def read_frame(self):
        self.ret, frame = self.cap.read()
        self.timestamps.append(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3)
        if self.ret:
            xr, yr = self.params['xr'], self.params['yr']
            self.cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[xr[0]:xr[1], yr[0]:yr[1]]
        else:
            print('Warning: frame not found')

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        print(f'x={ix}, y={iy}')
        self.clicked_coordinate.append([ix, iy])

    def photocheck(self):
        while self.params['check']:
            print(f'current N={len(self.circles)}')
            self.clicked_coordinate = []
            fig, ax = self.photo_plot()
            fig.canvas.mpl_connect('button_press_event', self.onclick)
            plt.show()

            if len(self.clicked_coordinate) == 0:
                break

            for unit in self.clicked_coordinate:
                y, x = int(unit[1]), int(unit[0])
                ROIr = self.params['r_ROI']

                x_min = max(x - ROIr, 0)
                x_max = min(x + ROIr, self.cimg.shape[1])
                y_min = max(y - ROIr, 0)
                y_max = min(y + ROIr, self.cimg.shape[0])

                self.ROIimg = self.cimg[y_min:y_max, x_min:x_max]

                while True:
                    added_circle = self.circlefitter(self.ROIimg)
                    if len(added_circle) == 0:
                        self.params['p2'] -= 1
                    elif len(added_circle) > 1:
                        self.params['p2'] += 1
                    else:
                        break

                added_circle[0][:2] += [x_min, y_min]
                added_circle = np.array(added_circle[0])
                try:
                    self.circles = np.append(self.circles, [added_circle], axis=0)
                except ValueError:
                    self.circles = np.array([added_circle])

    def photo_plot(self):
        fig, ax = plt.subplots(1, figsize=(6, 6))
        circles = np.asarray(self.circles)
        ax.imshow(cv2.cvtColor(self.cimg, cv2.COLOR_GRAY2BGR))
        for n, i in enumerate(circles):
            x, y, r = i
            ax.annotate(str(n), [x, y], c='white')
            ax.add_patch(plt.Circle((x, y), r, color='r', lw=1, fill=False))
        return fig, ax

    def circlefitter(self, cimg):
        p1, p2 = self.params['p1'], self.params['p2']
        mindist = self.params['MinDist']
        r1, r2 = self.params['r_obj']
        try:
            circles = cv2.HoughCircles(
                cimg, cv2.HOUGH_GRADIENT, 1, mindist,
                param1=p1, param2=p2, minRadius=r1, maxRadius=r2
            )
            if circles is not None:
                circles = np.around(circles[0, :]).astype(np.float32)  # convert to float to avoid uint16 math issues
            else:
                circles = []
        except cv2.error:
            circles = []
        return circles

    def manual_detection(self):
        """Manual click for a single particle center when auto-detection fails."""
        self.clicked_coordinate = []
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(cv2.cvtColor(self.ROIimg, cv2.COLOR_GRAY2BGR))
        ax.set_title("Click the particle center, then close window")
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

        if len(self.clicked_coordinate) == 0:
            print("No click detected — aborting")
            self.abort()

        xy = self.clicked_coordinate[0]
        # always return [x, y, r]
        xy = [float(xy[0]), float(xy[1]), float(self.params['r_obj'][0])]  # assign radius ~ expected
        return np.array([xy], dtype=np.float32)  # always 2D (1,3)

    def abort(self):
        self.savedata()
        print('Tracking aborted but data saved.')
        sys.exit()

    def increase_threshold(self):
        self.params['p2'] += 1
        print(f'p2 increased to {self.params["p2"]}')

    def decrease_threshold(self):
        self.params['p2'] -= 1
        print(f'p2 decreased to {self.params["p2"]}')

    def extract_data(self):
        if not os.path.exists(self.picklepath):
            os.makedirs(self.picklepath)
        output_path = os.path.join(self.picklepath, self.filename)
        if not os.path.isfile(output_path) or self.overwrite:
            data = [self.data.raw, self.data.timestamps]
            pickle.dump(data, open(output_path, 'wb'))
        else:
            print(f"Data not saved (overwrite=False and {self.filename} exists)")

    def load_data(self):
        with open(os.path.join(self.picklepath, self.filename), 'rb') as openfile:
            try:
                raw, timestamps = pickle.load(openfile)
                return raw, timestamps
            except EOFError:
                print('No such file')
                return None, None

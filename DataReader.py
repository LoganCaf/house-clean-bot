# Various ways just to visualize the image


# from svglib.svglib import svg2rlg
# def is_svg_by_svglib(filepath):
#     try:
#         drawing = svg2rlg(filepath)
#         return True
#     except Exception:
#         return False
    

# from IPython.display import SVG, display

# svg_data = 'train-00/0000-0003.svg'
# display(SVG(svg_data))


# import sys
# from PyQt5 import QtSvg
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtSvg import QSvgWidget

# app = QApplication(sys.argv)
# svg_widget = QtSvg.QSvgWidget('train-00/0000-0003.svg')
# svg_widget.show()
# sys.exit(app.exec_())








# Using svgpathtools to parse the image

from svgpathtools import svg2paths2
import numpy as np
import re

def svg_to_color_grid(svg_file, grid_size=(300,300)):
    # 1) Read all <path>‑like shapes (it should convert rect/circle/poly/etc. to paths)
    paths, attributes, svg_attribs = svg2paths2(svg_file)

    # 2) Determine coordinate space
    viewBox = svg_attribs.get('viewBox')
    if viewBox:
        _, _, w, h = map(float, viewBox.split())
    else:
        w = float(svg_attribs.get('width'))
        h = float(svg_attribs.get('height'))

    rows, cols = grid_size
    color_grid = np.ones((rows, cols, 3), dtype=float)

    def to_grid(pt):
        x, y = pt.real, pt.imag
        i = int((y/h)*rows)
        j = int((x/w)*cols)
        return i, j

    def parse_color(col):
        if not col or col.lower() == 'none':
            return (0, 0, 0)
        col = col.strip()
        # Hex format
        if col.startswith('#'):
            h = col.lstrip('#')
            if len(h) == 3:
                h = ''.join(2*c for c in h)
            if len(h) == 6:
                try:
                    return tuple(int(h[i:i+2], 16)/255 for i in (0,2,4))
                except ValueError:
                    pass

        # rgb() or rgba()
        m = re.match(r'rgba?\(\s*([0-9]+),\s*([0-9]+),\s*([0-9]+)', col)
        if m:
            try:
                r, g, b = map(int, m.groups())
                return (r/255, g/255, b/255)
            except:
                pass

        # 3) Named colors?
        named = {
        'black': (0,0,0),
        'white': (1,1,1),
        'red':   (1,0,0),
        'blue':  (0,0,1),
        }   # can probably add more
        return named.get(col.lower(), (0,0,0))

    # 3) For each path segment, sample N points along it
    for path, attr in zip(paths, attributes):
        # pick up stroke/fill
        col = attr.get('stroke') or attr.get('fill') or 'black'
        rgb = parse_color(col)

        # sample each segment at 20 points for now
        for seg in path:
            for t in np.linspace(0, 1, 20):
                i, j = to_grid(seg.point(t))
                if 0 <= i < rows and 0 <= j < cols:
                    color_grid[i,j,:] = rgb

    return color_grid

def svg_to_binary_grid(svg_file, grid_size=(300,300)):
    rgb = svg_to_color_grid(svg_file, grid_size=grid_size)  # (H,W,3)
    # anything not pure white background → wall
    binary = (rgb.min(axis=2) < 0.99).astype(np.uint8)
    return binary






####### Visualize the data

import matplotlib.pyplot as plt

# grid, ... =
# color_grid = svg_to_color_grid('train-00/0000-0003.svg')
#grid = binary_dilation(grid, iterations=1).astype(np.uint8)

# plt.figure(figsize=(6,6))
# plt.imshow(color_grid, interpolation='nearest')     # change to 'nearest' for blockier image but more precise, bilinear for smoother
# plt.axis('off')
# plt.title("Parsed Grid")
# plt.show()


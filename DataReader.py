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










### Parsing SVG for data recognition ###


import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import binary_dilation
from math import cos, sin, radians, pi


color_grid = None

import re

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
    }
    return named.get(col.lower(), (0,0,0))


def bezier_curve(points, num_points=20):
    # Handles bezier curves in SVG files
    if len(points) == 3:  # Quadratic
        P0, P1, P2 = points
        return [( (1-t)**2*P0[0] + 2*(1-t)*t*P1[0] + t**2*P2[0],
                  (1-t)**2*P0[1] + 2*(1-t)*t*P1[1] + t**2*P2[1])
                for t in np.linspace(0, 1, num_points)]

    elif len(points) == 4:  # Cubic
        P0, P1, P2, P3 = points
        return [( (1-t)**3*P0[0] + 3*(1-t)**2*t*P1[0] + 3*(1-t)*t**2*P2[0] + t**3*P3[0],
                  (1-t)**3*P0[1] + 3*(1-t)**2*t*P1[1] + 3*(1-t)*t**2*P2[1] + t**3*P3[1])
                for t in np.linspace(0, 1, num_points)]
    return []

def approximate_arc(start, end, rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, num_points=20):
    # Simplified arc approximation: assumes small arcs and ignores rotation for now
    x0, y0 = start
    x1, y1 = end

    # Just sample evenly along an ellipse between start and end
    theta_start = 0
    theta_end = pi if large_arc_flag else pi / 2
    if not sweep_flag:
        theta_start, theta_end = theta_end, theta_start

    return [
        (
            (x0 + x1)/2 + rx * cos(t) * cos(radians(x_axis_rotation)),
            (y0 + y1)/2 + ry * sin(t) * sin(radians(x_axis_rotation))
        )
        for t in np.linspace(theta_start, theta_end, num_points)
    ]

def parse_path(d_str, toGrid, num_samples=20):
    commands = []
    tokens = d_str.replace(',', ' ').split()
    i = 0
    current_cmd = None
    cursor = (0, 0)

    def is_command(token):
        return token in {'M', 'L', 'C', 'Q', 'Z', 'A', 'a', 'H', 'h', 'V', 'v', 'S', 's', 'T', 't'}

    def parse_point(index):
        return float(tokens[index]), float(tokens[index + 1])

    while i < len(tokens):
        token = tokens[i]

        if is_command(token):
            current_cmd = token
            i += 1
            continue

        # Skip unsupported commands
        if current_cmd in {'a', 'H', 'h', 'V', 'v', 'S', 's', 'T', 't'}:
            i += 1
            continue

        try:
            if current_cmd == 'M':
                x, y = parse_point(i)
                cursor = (x, y)
                r, c = toGrid(x, y)
                commands.append(('M', r, c))
                i += 2

            elif current_cmd == 'L':
                x, y = parse_point(i)
                cursor = (x, y)
                r, c = toGrid(x, y)
                commands.append(('L', r, c))
                i += 2

            elif current_cmd == 'Q':
                x1, y1 = parse_point(i)
                x2, y2 = parse_point(i + 2)
                bezier_pts = bezier_curve([cursor, (x1, y1), (x2, y2)], num_samples)
                for pt in bezier_pts:
                    r, c = toGrid(*pt)
                    commands.append(('L', r, c))
                cursor = (x2, y2)
                i += 4

            elif current_cmd == 'A':
                rx = float(tokens[i])
                ry = float(tokens[i+1])
                x_axis_rotation = float(tokens[i+2])
                large_arc_flag = int(tokens[i+3])
                sweep_flag = int(tokens[i+4])
                x, y = float(tokens[i+5]), float(tokens[i+6])

                arc_points = approximate_arc(cursor, (x, y), rx, ry, x_axis_rotation, large_arc_flag, sweep_flag)

                for pt in arc_points:
                    r, c = toGrid(*pt)
                    commands.append(('L', r, c))

                cursor = (x, y)
                i += 7

            elif current_cmd == 'C':
                x1, y1 = parse_point(i)
                x2, y2 = parse_point(i + 2)
                x3, y3 = parse_point(i + 4)
                bezier_pts = bezier_curve([cursor, (x1, y1), (x2, y2), (x3, y3)], num_samples)
                for pt in bezier_pts:
                    r, c = toGrid(*pt)
                    commands.append(('L', r, c))
                cursor = (x3, y3)
                i += 6

            elif current_cmd == 'Z':
                commands.append(('Z',))
                i += 1

            else:
                # Implicit "L"
                x, y = parse_point(i)
                cursor = (x, y)
                r, c = toGrid(x, y)
                commands.append(('L', r, c))
                i += 2

        except (IndexError, ValueError):
            i += 1  # Skip broken or partial command and try to recover

    return commands




def parseSVG(svg_path, grid_size=(300, 300), canvas_size=(500, 500)):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg':'http://www.w3.org/2000/svg'}     # Supposedly this is necessary 

    grid = np.zeros(grid_size, dtype=np.uint8)
    rows, cols = grid_size
    width, height = canvas_size

    color_grid = np.ones((rows, cols, 3), dtype=float)  # default white background

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

    def toGrid(x, y):
        gx = int((x / width) * cols)
        gy = int((y / height) * rows)
        return gy, gx
    

    print("SVG width/height:", root.attrib.get('width'), root.attrib.get('height'))
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        print("Found tag:", tag)

    viewBox = root.attrib.get("viewBox")
    if viewBox:
        _, _, w, h = map(float, viewBox.strip().split())
        canvas_size = (w, h)


    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        if tag == 'rect':
            x = float(elem.attrib.get('x', 0))
            y = float(elem.attrib.get('y', 0))
            w = float(elem.attrib.get('width', 0))
            h = float(elem.attrib.get('height', 0))

            top_left = toGrid(x, y)
            bottom_right = toGrid(x + w, y + h)
            # For colors
            col = elem.attrib.get('stroke') or elem.attrib.get('fill') or None
            rgb = parse_color(col)
            for i in range(top_left[0], bottom_right[0]):
                for j in range(top_left[1], bottom_right[1]):
                    if 0 <= i < rows and 0 <= j < cols:
                        grid[i][j] = 1
                        color_grid[i, j, :] = rgb

        elif tag == 'line':
            x1 = float(elem.attrib.get('x1', 0))
            y1 = float(elem.attrib.get('y1', 0))
            x2 = float(elem.attrib.get('x2', 0))
            y2 = float(elem.attrib.get('y2', 0))
            
            # Get grid coordinates
            r1, c1 = toGrid(x1, y1)
            r2, c2 = toGrid(x2, y2)

            # Draw a simple line
            num_steps = max(abs(r2 - r1), abs(c2 - c1)) + 1
            # For colors
            col = elem.attrib.get('stroke') or elem.attrib.get('fill') or None
            rgb = parse_color(col)
            for step in range(num_steps):
                i = int(r1 + (r2 - r1) * step / num_steps)
                j = int(c1 + (c2 - c1) * step / num_steps)
                if 0 <= i < rows and 0 <= j < cols:
                    grid[i][j] = 1
                    color_grid[i, j, :] = rgb
        elif tag == 'path':
            d = elem.attrib.get('d', '')
            commands = parse_path(d, toGrid)

            points = []
            for cmd in commands:
                if cmd[0] in {'M', 'L'}:
                    points.append((cmd[1], cmd[2]))

            if commands and commands[-1][0] == 'Z' and len(points) > 2:
                points.append(points[0])

            # For colors
            col = elem.attrib.get('stroke') or elem.attrib.get('fill') or None
            rgb = parse_color(col)
            for (r1, c1), (r2, c2) in zip(points, points[1:]):
                steps = max(abs(r2 - r1), abs(c2 - c1)) + 1
                for step in range(steps):
                    i = int(r1 + (r2 - r1) * step / steps)
                    j = int(c1 + (c2 - c1) * step / steps)
                    if 0 <= i < rows and 0 <= j < cols:
                        grid[i][j] = 1
                        color_grid[i, j, :] = rgb


        elif tag == 'circle':
            cx = float(elem.attrib.get('cx', 0))
            cy = float(elem.attrib.get('cy', 0))
            r = float(elem.attrib.get('r', 0))

            center_r, center_c = toGrid(cx, cy)
            pixel_radius_r = int((r / height) * rows)
            pixel_radius_c = int((r / width) * cols)

            # For colors
            col = elem.attrib.get('stroke') or elem.attrib.get('fill') or None
            rgb = parse_color(col)
            for i in range(center_r - pixel_radius_r, center_r + pixel_radius_r + 1):
                for j in range(center_c - pixel_radius_c, center_c + pixel_radius_c + 1):
                    if 0 <= i < rows and 0 <= j < cols:
                        # Check if inside the circle
                        if ((i - center_r)**2) / (pixel_radius_r**2 + 1e-5) + ((j - center_c)**2) / (pixel_radius_c**2 + 1e-5) <= 1:
                            grid[i][j] = 1
                            color_grid[i, j, :] = rgb

        elif tag == 'polygon':
            point_str = elem.attrib.get('points', '')
            points = []
            for pair in point_str.replace(',', ' ').split():
                try:
                    x, y = map(float, pair.strip().split())
                    r, c = toGrid(x, y)
                    points.append((r, c))
                except:
                    continue

            # For colors -- this may be better placed somewhere else in this particular if-statement
            col = elem.attrib.get('stroke') or elem.attrib.get('fill') or None
            rgb = parse_color(col)
            # Connect all pairs of points
            for (r1, c1), (r2, c2) in zip(points, points[1:] + [points[0]]):  # auto-close polygon
                steps = max(abs(r2 - r1), abs(c2 - c1)) + 1
                for step in range(steps):
                    i = int(r1 + (r2 - r1) * step / steps)
                    j = int(c1 + (c2 - c1) * step / steps)
                    if 0 <= i < rows and 0 <= j < cols:
                        grid[i][j] = 1
                        color_grid[i, j, :] = rgb

        elif tag == 'ellipse':
            cx = float(elem.attrib.get('cx', 0))
            cy = float(elem.attrib.get('cy', 0))
            rx = float(elem.attrib.get('rx', 0))
            ry = float(elem.attrib.get('ry', 0))

            center_r, center_c = toGrid(cx, cy)
            pixel_rx = int((rx / width) * cols)
            pixel_ry = int((ry / height) * rows)

            # For colors
            col = elem.attrib.get('stroke') or elem.attrib.get('fill') or None
            rgb = parse_color(col)
            for i in range(center_r - pixel_ry, center_r + pixel_ry + 1):
                for j in range(center_c - pixel_rx, center_c + pixel_rx + 1):
                    if 0 <= i < rows and 0 <= j < cols:
                        # Ellipse formula
                        if ((i - center_r)**2) / (pixel_ry**2 + 1e-5) + ((j - center_c)**2) / (pixel_rx**2 + 1e-5) <= 1:
                            grid[i][j] = 1
                            color_grid[i, j, :] = rgb


    # If this prints 0, map is being rendered empty. Either shapes are being skipped or grid conversion isn't matching SVG space
    print("Nonzero entries in grid:", np.count_nonzero(grid))
    print("SVG width/height:", root.attrib.get('width'), root.attrib.get('height'))
    print("viewBox: ", viewBox)


    return grid, color_grid

import matplotlib.pyplot as plt

grid, color_grid = parseSVG('train-00/0000-0003.svg')
grid = binary_dilation(grid, iterations=1).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(color_grid, interpolation='nearest')     # change to 'nearest' for blockier image but more precise, bilinear for smoother
plt.axis('off')
plt.title("Parsed Grid")
plt.show()


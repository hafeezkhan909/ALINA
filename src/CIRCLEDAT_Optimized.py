from itertools import product

def circular_threshold_pixel_discovery_and_traversal(img, x, y, threshold, visited, line_marking_pixels):
    stack = [(x, y)]
    visited.add((x, y))

    circular_range = list(product(range(-threshold, threshold+1), repeat=2))
    circular_range.remove((0, 0))

    while stack:
        x, y = stack.pop()

        if img[y][x] == 255:
            line_marking_pixels.append((x, y))

            for i, j in circular_range:
                x_new = x + i
                y_new = y + j

                if (x_new, y_new) in visited:
                    continue

                if x_new < 0 or y_new < 0 or x_new >= img.shape[1] or y_new >= img.shape[0]:
                    continue

                stack.append((x_new, y_new))
                visited.add((x_new, y_new))

    return line_marking_pixels


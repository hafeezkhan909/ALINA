def circular_threshold_pixel_discovery_and_traversal(img, x, y, threshold, visited, line_marking_pixels):
    stack = [(x, y)]
    visited[y][x] = 1

    while len(stack) > 0:
        x, y = stack.pop()

        if img[y][x] == 255:
            line_marking_pixels.append((x, y))
            #print(line_marking_pixels)

            # Check neighboring pixels within the circular threshold
            for i in range(-threshold, threshold + 1):
                for j in range(-threshold, threshold + 1):
                    if i == 0 and j == 0:
                        continue

                    x_new = x + i
                    y_new = y + j

                    if x_new < 0 or y_new < 0 or x_new >= img.shape[1] or y_new >= img.shape[0]:
                        continue

                    if visited[y_new][x_new] == 0:
                        stack.append((x_new, y_new))
                        visited[y_new][x_new] = 1
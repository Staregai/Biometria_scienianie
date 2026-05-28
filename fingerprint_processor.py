from __future__ import annotations

from dataclasses import dataclass
import math
from collections import deque
from typing import Iterable

from PIL import Image, ImageDraw


K3M_A0 = {
    3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120, 124,
    126, 127, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224,
    225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254,
}
K3M_A1 = {7, 14, 28, 56, 112, 131, 193, 224}
K3M_A2 = {7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240}
K3M_A3 = {
    7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135, 143, 193, 195,
    199, 224, 225, 227, 240, 241, 248,
}
K3M_A4 = {
    7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143,
    159, 193, 195, 199, 207, 224, 225, 227, 231, 240, 241, 243, 248, 249, 252,
}
K3M_A5 = {
    7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143,
    159, 191, 193, 195, 199, 207, 224, 225, 227, 231, 239, 240, 241, 243, 248,
    249, 251, 252, 254,
}
K3M_A1PIX = K3M_A0.copy()
K3M_PHASES = [K3M_A1, K3M_A2, K3M_A3, K3M_A4, K3M_A5]
NEIGHBOR_BITS = (
    (-1, -1, 128),
    (0, -1, 1),
    (1, -1, 2),
    (1, 0, 4),
    (1, 1, 8),
    (0, 1, 16),
    (-1, 1, 32),
    (-1, 0, 64),
)
KMM_DELETION_WEIGHTS = {
    3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48, 52, 53, 54,
    55, 56, 60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80, 81, 83, 84, 85, 86,
    87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101, 103, 109, 111, 112, 113, 115,
    116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 131, 133, 135, 141,
    143, 149, 151, 157, 159, 181, 183, 189, 191, 192, 193, 195, 197, 199, 205,
    207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223,
    224, 225, 227, 229, 231, 237, 239, 240, 241, 243, 244, 245, 246, 247, 248,
    249, 251, 252, 253, 254, 255,
}


def clamp(value: float, minimum: int = 0, maximum: int = 255) -> int:
    rounded = int(round(value))
    if rounded < minimum:
        return minimum
    if rounded > maximum:
        return maximum
    return rounded


@dataclass
class GrayImage:
    width: int
    height: int
    pixels: list[list[int]]

    def copy(self) -> "GrayImage":
        return GrayImage(self.width, self.height, [row[:] for row in self.pixels])


@dataclass
class BinaryImage:
    width: int
    height: int
    pixels: list[list[int]]

    def copy(self) -> "BinaryImage":
        return BinaryImage(self.width, self.height, [row[:] for row in self.pixels])

    def count_foreground(self) -> int:
        return sum(sum(row) for row in self.pixels)


@dataclass
class Component:
    area: int
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    center_x: float
    center_y: float
    pixels: list[tuple[int, int]]


@dataclass
class Minutia:
    x: int
    y: int
    kind: str


@dataclass
class AlgorithmResult:
    skeleton_raw: BinaryImage
    skeleton_final: BinaryImage
    minutiae: list[Minutia]
    overlay: Image.Image
    summary: str


@dataclass
class FingerprintAnalysis:
    source: GrayImage
    enhanced: GrayImage
    binary: BinaryImage
    cleaned: BinaryImage
    threshold: int
    morph: AlgorithmResult
    k3m: AlgorithmResult
    kmm: AlgorithmResult
    comparison_summary: str


def from_pillow(image: Image.Image) -> GrayImage:
    gray = image.convert("L")
    width, height = gray.size
    data = list(gray.getdata())
    pixels = [data[y * width : (y + 1) * width] for y in range(height)]
    return GrayImage(width, height, pixels)


def gray_to_pillow(image: GrayImage) -> Image.Image:
    output = Image.new("L", (image.width, image.height))
    output.putdata([value for row in image.pixels for value in row])
    return output


def binary_to_gray(binary: BinaryImage, foreground: int = 0, background: int = 255) -> GrayImage:
    pixels = []
    for row in binary.pixels:
        pixels.append([foreground if value else background for value in row])
    return GrayImage(binary.width, binary.height, pixels)


def histogram(image: GrayImage) -> list[int]:
    hist = [0] * 256
    for row in image.pixels:
        for value in row:
            hist[value] += 1
    return hist


def is_binary_like(image: GrayImage, max_unique: int = 4, min_separation: int = 32) -> bool:
    if max_unique < 2:
        raise ValueError("max_unique must be >= 2.")
    if min_separation < 0:
        raise ValueError("min_separation must be >= 0.")
    hist = histogram(image)
    unique = [value for value, count in enumerate(hist) if count]
    if len(unique) > max_unique:
        return False
    if len(unique) <= 1:
        return True
    return unique[-1] - unique[0] >= min_separation


def percentile_value(image: GrayImage, percentile: float) -> int:
    hist = histogram(image)
    total = image.width * image.height
    if total == 0:
        return 0
    target = int(round((min(max(percentile, 0.0), 100.0) / 100.0) * (total - 1)))
    cumulative = 0
    for value, count in enumerate(hist):
        cumulative += count
        if cumulative > target:
            return value
    return 255


def contrast_stretch(image: GrayImage, low_percentile: float = 2.0, high_percentile: float = 98.0) -> GrayImage:
    low = percentile_value(image, low_percentile)
    high = percentile_value(image, high_percentile)
    if high <= low:
        return image.copy()
    scale = 255.0 / (high - low)
    pixels: list[list[int]] = []
    for row in image.pixels:
        pixels.append([clamp((value - low) * scale) for value in row])
    return GrayImage(image.width, image.height, pixels)


def otsu_threshold(image: GrayImage) -> int:
    hist = histogram(image)
    total = image.width * image.height
    if total == 0:
        return 0
    total_sum = sum(index * count for index, count in enumerate(hist))
    sum_background = 0
    weight_background = 0
    max_variance = -1.0
    best_threshold = 0

    for threshold, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (total_sum - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    return best_threshold


def _integral_image(values: list[list[int]]) -> list[list[int]]:
    height = len(values)
    width = len(values[0]) if height else 0
    integral = [[0 for _ in range(width + 1)] for _ in range(height + 1)]
    for y in range(1, height + 1):
        row_sum = 0
        src_row = values[y - 1]
        for x in range(1, width + 1):
            row_sum += src_row[x - 1]
            integral[y][x] = integral[y - 1][x] + row_sum
    return integral


def _window_sum(integral: list[list[int]], x0: int, y0: int, x1: int, y1: int) -> int:
    return (
        integral[y1 + 1][x1 + 1]
        - integral[y0][x1 + 1]
        - integral[y1 + 1][x0]
        + integral[y0][x0]
    )


def sauvola_threshold_map(
    image: GrayImage,
    window_size: int = 21,
    k: float = 0.2,
    dynamic_range: float = 128.0,
) -> list[list[float]]:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Rozmiar okna Sauvoli musi byc nieparzysty i >= 3.")
    if not (0.0 <= k <= 1.0):
        raise ValueError("Parametr k musi byc w zakresie [0, 1].")
    if dynamic_range <= 0.0:
        raise ValueError("Dynamiczny zakres musi byc > 0.")

    height = image.height
    width = image.width
    pixels = image.pixels
    integral = _integral_image(pixels)
    squared = [[value * value for value in row] for row in pixels]
    integral_sq = _integral_image(squared)
    radius = window_size // 2
    threshold_map: list[list[float]] = []

    for y in range(height):
        row: list[float] = []
        y0 = max(0, y - radius)
        y1 = min(height - 1, y + radius)
        for x in range(width):
            x0 = max(0, x - radius)
            x1 = min(width - 1, x + radius)
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            total = _window_sum(integral, x0, y0, x1, y1)
            total_sq = _window_sum(integral_sq, x0, y0, x1, y1)
            mean = total / area
            variance = max((total_sq / area) - (mean * mean), 0.0)
            std = math.sqrt(variance)
            threshold = mean * (1.0 + k * ((std / dynamic_range) - 1.0))
            row.append(threshold)
        threshold_map.append(row)
    return threshold_map


def binarize_fingerprint(
    image: GrayImage,
    method: str = "otsu",
    threshold: int | None = None,
    invert: bool = False,
    sauvola_window_size: int = 21,
    sauvola_k: float = 0.2,
    sauvola_dynamic_range: float = 128.0,
) -> tuple[BinaryImage, int]:
    if threshold is not None:
        level = clamp(threshold)
        binary = threshold_dark_ridges(image, level)
    elif method == "auto":
        if is_binary_like(image):
            min_value = min(min(row) for row in image.pixels)
            max_value = max(max(row) for row in image.pixels)
            level = clamp((min_value + max_value) / 2.0)
            binary = threshold_dark_ridges(image, level)
        else:
            thresholds = sauvola_threshold_map(
                image,
                window_size=sauvola_window_size,
                k=sauvola_k,
                dynamic_range=sauvola_dynamic_range,
            )
            binary_pixels: list[list[int]] = []
            total_threshold = 0.0
            for y, row in enumerate(image.pixels):
                binary_row: list[int] = []
                for x, value in enumerate(row):
                    threshold_value = thresholds[y][x]
                    total_threshold += threshold_value
                    binary_row.append(1 if value <= threshold_value else 0)
                binary_pixels.append(binary_row)
            level = clamp(total_threshold / (image.width * image.height))
            binary = BinaryImage(image.width, image.height, binary_pixels)
    elif method == "sauvola":
        thresholds = sauvola_threshold_map(
            image,
            window_size=sauvola_window_size,
            k=sauvola_k,
            dynamic_range=sauvola_dynamic_range,
        )
        binary_pixels = []
        total_threshold = 0.0
        for y, row in enumerate(image.pixels):
            binary_row: list[int] = []
            for x, value in enumerate(row):
                threshold_value = thresholds[y][x]
                total_threshold += threshold_value
                binary_row.append(1 if value <= threshold_value else 0)
            binary_pixels.append(binary_row)
        level = clamp(total_threshold / (image.width * image.height))
        binary = BinaryImage(image.width, image.height, binary_pixels)
    elif method == "otsu":
        level = otsu_threshold(image)
        binary = threshold_dark_ridges(image, level)
    else:
        raise ValueError(f"Nieznana metoda binarizacji: {method}")

    if invert:
        binary = BinaryImage(image.width, image.height, [[0 if v else 1 for v in row] for row in binary.pixels])
    return binary, level


def threshold_dark_ridges(image: GrayImage, threshold: int) -> BinaryImage:
    pixels: list[list[int]] = []
    for row in image.pixels:
        pixels.append([1 if value <= threshold else 0 for value in row])
    return BinaryImage(image.width, image.height, pixels)


def estimate_fingerprint_mask(
    image: GrayImage,
    window_size: int = 17,
    min_std: float = 12.0,
    kernel_size: int = 9,
) -> BinaryImage:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Rozmiar okna maski musi byc nieparzysty i >= 3.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Rozmiar elementu maski musi byc nieparzysty i >= 1.")
    if min_std < 0.0:
        raise ValueError("Minimalne odchylenie standardowe musi byc >= 0.")

    height = image.height
    width = image.width
    pixels = image.pixels
    integral = _integral_image(pixels)
    squared = [[value * value for value in row] for row in pixels]
    integral_sq = _integral_image(squared)
    radius = window_size // 2

    mask_pixels: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        y0 = max(0, y - radius)
        y1 = min(height - 1, y + radius)
        for x in range(width):
            x0 = max(0, x - radius)
            x1 = min(width - 1, x + radius)
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            total = _window_sum(integral, x0, y0, x1, y1)
            total_sq = _window_sum(integral_sq, x0, y0, x1, y1)
            mean = total / area
            variance = max((total_sq / area) - (mean * mean), 0.0)
            std = math.sqrt(variance)
            row.append(1 if std >= min_std else 0)
        mask_pixels.append(row)

    mask = BinaryImage(width, height, mask_pixels)
    if kernel_size > 1:
        mask = morph_close(mask, kernel_size)
        mask = morph_open(mask, kernel_size)
    return mask


def _morphology_indices(length: int, size: int) -> list[list[int]]:
    radius = size // 2
    return [[min(max(index + shift - radius, 0), length - 1) for shift in range(size)] for index in range(length)]


def _validate_size(size: int) -> None:
    if size < 1 or size % 2 == 0:
        raise ValueError("Rozmiar elementu strukturalnego musi byc nieparzysty i dodatni.")


def dilate(binary: BinaryImage, size: int = 3) -> BinaryImage:
    _validate_size(size)
    y_indices = _morphology_indices(binary.height, size)
    x_indices = _morphology_indices(binary.width, size)
    pixels: list[list[int]] = []
    for y in range(binary.height):
        row: list[int] = []
        for x in range(binary.width):
            value = 0
            for sy in y_indices[y]:
                source_row = binary.pixels[sy]
                for sx in x_indices[x]:
                    if source_row[sx]:
                        value = 1
                        break
                if value:
                    break
            row.append(value)
        pixels.append(row)
    return BinaryImage(binary.width, binary.height, pixels)


def erode(binary: BinaryImage, size: int = 3) -> BinaryImage:
    _validate_size(size)
    y_indices = _morphology_indices(binary.height, size)
    x_indices = _morphology_indices(binary.width, size)
    pixels: list[list[int]] = []
    for y in range(binary.height):
        row: list[int] = []
        for x in range(binary.width):
            value = 1
            for sy in y_indices[y]:
                source_row = binary.pixels[sy]
                for sx in x_indices[x]:
                    if not source_row[sx]:
                        value = 0
                        break
                if not value:
                    break
            row.append(value)
        pixels.append(row)
    return BinaryImage(binary.width, binary.height, pixels)


def morph_open(binary: BinaryImage, size: int = 3) -> BinaryImage:
    return dilate(erode(binary, size), size)


def morph_close(binary: BinaryImage, size: int = 3) -> BinaryImage:
    return erode(dilate(binary, size), size)


def and_not(image_a: BinaryImage, image_b: BinaryImage) -> BinaryImage:
    pixels: list[list[int]] = []
    for row_a, row_b in zip(image_a.pixels, image_b.pixels):
        pixels.append([1 if value_a and not value_b else 0 for value_a, value_b in zip(row_a, row_b)])
    return BinaryImage(image_a.width, image_a.height, pixels)


def binary_or(image_a: BinaryImage, image_b: BinaryImage) -> BinaryImage:
    pixels: list[list[int]] = []
    for row_a, row_b in zip(image_a.pixels, image_b.pixels):
        pixels.append([1 if value_a or value_b else 0 for value_a, value_b in zip(row_a, row_b)])
    return BinaryImage(image_a.width, image_a.height, pixels)


def binary_and(image_a: BinaryImage, image_b: BinaryImage) -> BinaryImage:
    pixels: list[list[int]] = []
    for row_a, row_b in zip(image_a.pixels, image_b.pixels):
        pixels.append([1 if value_a and value_b else 0 for value_a, value_b in zip(row_a, row_b)])
    return BinaryImage(image_a.width, image_a.height, pixels)


def connected_components(binary: BinaryImage) -> list[Component]:
    visited = [[False for _ in range(binary.width)] for _ in range(binary.height)]
    components: list[Component] = []

    for y in range(binary.height):
        for x in range(binary.width):
            if not binary.pixels[y][x] or visited[y][x]:
                continue

            queue_points: deque[tuple[int, int]] = deque([(x, y)])
            visited[y][x] = True
            points: list[tuple[int, int]] = []

            while queue_points:
                px, py = queue_points.popleft()
                points.append((px, py))
                for ny in range(max(py - 1, 0), min(py + 2, binary.height)):
                    for nx in range(max(px - 1, 0), min(px + 2, binary.width)):
                        if binary.pixels[ny][nx] and not visited[ny][nx]:
                            visited[ny][nx] = True
                            queue_points.append((nx, ny))

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            components.append(
                Component(
                    area=len(points),
                    min_x=min(xs),
                    min_y=min(ys),
                    max_x=max(xs),
                    max_y=max(ys),
                    center_x=sum(xs) / len(xs),
                    center_y=sum(ys) / len(ys),
                    pixels=points,
                )
            )

    return components


def largest_central_component(binary: BinaryImage, min_area: int = 300) -> BinaryImage:
    components = [component for component in connected_components(binary) if component.area >= min_area]
    if not components:
        return binary.copy()

    center_x = binary.width / 2.0
    center_y = binary.height / 2.0

    def score(component: Component) -> float:
        distance = math.hypot(component.center_x - center_x, component.center_y - center_y)
        return component.area - 2.5 * distance

    chosen = max(components, key=score)
    active = set(chosen.pixels)
    pixels: list[list[int]] = []
    for y in range(binary.height):
        row: list[int] = []
        for x in range(binary.width):
            row.append(1 if (x, y) in active else 0)
        pixels.append(row)
    return BinaryImage(binary.width, binary.height, pixels)


def remove_small_components(binary: BinaryImage, min_area: int = 12) -> BinaryImage:
    kept_points: set[tuple[int, int]] = set()
    for component in connected_components(binary):
        if component.area >= min_area:
            kept_points.update(component.pixels)

    pixels: list[list[int]] = []
    for y in range(binary.height):
        row: list[int] = []
        for x in range(binary.width):
            row.append(1 if (x, y) in kept_points else 0)
        pixels.append(row)
    return BinaryImage(binary.width, binary.height, pixels)


def bridge_gaps(binary: BinaryImage, iterations: int = 1) -> BinaryImage:
    current = binary.copy()
    for _ in range(max(iterations, 0)):
        updated = [row[:] for row in current.pixels]
        changed = False
        for y in range(1, current.height - 1):
            for x in range(1, current.width - 1):
                if current.pixels[y][x]:
                    continue

                n = current.pixels[y - 1][x]
                s = current.pixels[y + 1][x]
                e = current.pixels[y][x + 1]
                w = current.pixels[y][x - 1]
                ne = current.pixels[y - 1][x + 1]
                nw = current.pixels[y - 1][x - 1]
                se = current.pixels[y + 1][x + 1]
                sw = current.pixels[y + 1][x - 1]
                neighbors = n + s + e + w + ne + nw + se + sw

                should_fill = (
                    (n and s)
                    or (e and w)
                    or (ne and sw)
                    or (nw and se)
                    or (n and se and neighbors <= 3)
                    or (n and sw and neighbors <= 3)
                    or (s and ne and neighbors <= 3)
                    or (s and nw and neighbors <= 3)
                    or (e and nw and neighbors <= 3)
                    or (e and sw and neighbors <= 3)
                    or (w and ne and neighbors <= 3)
                    or (w and se and neighbors <= 3)
                )

                if should_fill:
                    updated[y][x] = 1
                    changed = True
        current = BinaryImage(current.width, current.height, updated)
        if not changed:
            break
    return current


def morphological_skeleton(binary: BinaryImage) -> BinaryImage:
    skeleton = BinaryImage(binary.width, binary.height, [[0 for _ in range(binary.width)] for _ in range(binary.height)])
    current = binary.copy()
    while current.count_foreground() > 0:
        eroded = erode(current, 3)
        opened = dilate(eroded, 3)
        residue = and_not(current, opened)
        skeleton = binary_or(skeleton, residue)
        current = eroded
    return skeleton


def _pixel_active(matrix: list[list[int]], x: int, y: int) -> int:
    return 1 if matrix[y][x] > 0 else 0


def neighbor_weight(matrix: list[list[int]], x: int, y: int) -> int:
    weight = 0
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    for dx, dy, bit in NEIGHBOR_BITS:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and matrix[ny][nx] > 0:
            weight += bit
    return weight


def thin_k3m(binary: BinaryImage, one_pixel_passes: int = 32) -> BinaryImage:
    matrix = [row[:] for row in binary.pixels]
    height = len(matrix)
    width = len(matrix[0]) if height else 0

    while True:
        for y in range(height):
            for x in range(width):
                if matrix[y][x] == 1 and neighbor_weight(matrix, x, y) in K3M_A0:
                    matrix[y][x] = 2

        iteration_changed = False
        for phase in K3M_PHASES:
            to_delete: list[tuple[int, int]] = []
            for y in range(height):
                for x in range(width):
                    if matrix[y][x] == 2 and neighbor_weight(matrix, x, y) in phase:
                        to_delete.append((x, y))
            if to_delete:
                iteration_changed = True
                for x, y in to_delete:
                    matrix[y][x] = 0

        for y in range(height):
            for x in range(width):
                if matrix[y][x] == 2:
                    matrix[y][x] = 1

        if not iteration_changed:
            break

    for _ in range(max(one_pixel_passes, 1)):
        to_delete = []
        for y in range(height):
            for x in range(width):
                if matrix[y][x] == 1 and neighbor_weight(matrix, x, y) in K3M_A1PIX:
                    to_delete.append((x, y))
        if not to_delete:
            break
        for x, y in to_delete:
            matrix[y][x] = 0

    return BinaryImage(binary.width, binary.height, matrix)


def thin_kmm(binary: BinaryImage, max_iterations: int | None = None) -> BinaryImage:
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive when provided.")

    matrix = [row[:] for row in binary.pixels]
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    iteration = 0

    while True:
        if max_iterations is not None and iteration >= max_iterations:
            break
        iteration += 1

        candidates: list[tuple[int, int]] = []
        for y in range(height):
            for x in range(width):
                if matrix[y][x] != 1:
                    continue
                if _kmm_neighbor_count(matrix, x, y) >= 8:
                    continue
                if _kmm_weight_at(matrix, x, y) in KMM_DELETION_WEIGHTS:
                    candidates.append((x, y))

        if not candidates:
            break

        removed = False
        for x, y in candidates:
            if matrix[y][x] != 1:
                continue
            if _kmm_is_deletable(matrix, x, y):
                matrix[y][x] = 0
                removed = True

        if not removed:
            break

    return BinaryImage(binary.width, binary.height, matrix)


def _ordered_neighbors(binary: BinaryImage, x: int, y: int) -> list[int]:
    return [
        binary.pixels[y - 1][x],
        binary.pixels[y - 1][x + 1],
        binary.pixels[y][x + 1],
        binary.pixels[y + 1][x + 1],
        binary.pixels[y + 1][x],
        binary.pixels[y + 1][x - 1],
        binary.pixels[y][x - 1],
        binary.pixels[y - 1][x - 1],
    ]


def _ordered_neighbors_matrix(matrix: list[list[int]], x: int, y: int) -> list[int]:
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    def get(nx: int, ny: int) -> int:
        if 0 <= nx < width and 0 <= ny < height and matrix[ny][nx] > 0:
            return 1
        return 0
    return [
        get(x, y - 1),
        get(x + 1, y - 1),
        get(x + 1, y),
        get(x + 1, y + 1),
        get(x, y + 1),
        get(x - 1, y + 1),
        get(x - 1, y),
        get(x - 1, y - 1),
    ]


def endpoints(binary: BinaryImage) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for y in range(1, binary.height - 1):
        for x in range(1, binary.width - 1):
            if not binary.pixels[y][x]:
                continue
            if sum(_ordered_neighbors(binary, x, y)) == 1:
                result.append((x, y))
    return result


def prune_spurs(binary: BinaryImage, iterations: int = 1) -> BinaryImage:
    current = binary.copy()
    for _ in range(max(iterations, 0)):
        to_delete = endpoints(current)
        if not to_delete:
            break
        for x, y in to_delete:
            current.pixels[y][x] = 0
    return current


def _neighbor_count(binary: BinaryImage, x: int, y: int) -> int:
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < binary.width and 0 <= ny < binary.height and binary.pixels[ny][nx]:
                count += 1
    return count


def _ridge_neighbors(binary: BinaryImage, x: int, y: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < binary.width and 0 <= ny < binary.height and binary.pixels[ny][nx]:
                neighbors.append((nx, ny))
    return neighbors


def _trace_spur(binary: BinaryImage, start: tuple[int, int], max_length: int) -> list[tuple[int, int]] | None:
    path: list[tuple[int, int]] = [start]
    previous: tuple[int, int] | None = None
    current = start

    while len(path) <= max_length:
        candidates = [neighbor for neighbor in _ridge_neighbors(binary, *current) if neighbor != previous]
        if not candidates:
            return path
        if len(candidates) > 1:
            return path
        next_pixel = candidates[0]
        if _neighbor_count(binary, *next_pixel) >= 3:
            return path
        path.append(next_pixel)
        previous = current
        current = next_pixel

    return None


def prune_short_spurs(binary: BinaryImage, max_length: int = 4, iterations: int = 1) -> BinaryImage:
    if max_length <= 0 or iterations <= 0:
        return binary.copy()
    current = binary.copy()

    for _ in range(iterations):
        to_remove: set[tuple[int, int]] = set()
        for x, y in endpoints(current):
            branch = _trace_spur(current, (x, y), max_length=max_length)
            if branch is not None and len(branch) <= max_length:
                to_remove.update(branch)
        if not to_remove:
            break
        for x, y in to_remove:
            current.pixels[y][x] = 0
    return current


def bounding_box(binary: BinaryImage) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for y, row in enumerate(binary.pixels):
        for x, value in enumerate(row):
            if value:
                xs.append(x)
                ys.append(y)
    if not xs:
        return (0, 0, binary.width - 1, binary.height - 1)
    return min(xs), min(ys), max(xs), max(ys)


def _crossing_number(neighbors: list[int]) -> int:
    total = 0
    loop = neighbors + [neighbors[0]]
    for first, second in zip(loop, loop[1:]):
        total += abs(first - second)
    return total // 2


def _kmm_neighbor_count(matrix: list[list[int]], x: int, y: int) -> int:
    return sum(_ordered_neighbors_matrix(matrix, x, y))


def _kmm_weight_at(matrix: list[list[int]], x: int, y: int) -> int:
    height = len(matrix)
    width = len(matrix[0]) if height else 0

    def get(nx: int, ny: int) -> int:
        if 0 <= nx < width and 0 <= ny < height and matrix[ny][nx] > 0:
            return 1
        return 0

    return (
        get(x, y - 1) * 1
        + get(x + 1, y - 1) * 2
        + get(x + 1, y) * 4
        + get(x + 1, y + 1) * 8
        + get(x, y + 1) * 16
        + get(x - 1, y + 1) * 32
        + get(x - 1, y) * 64
        + get(x - 1, y - 1) * 128
    )


def _kmm_touches_background(matrix: list[list[int]], x: int, y: int) -> bool:
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                return True
            if matrix[ny][nx] == 0:
                return True
    return False


def _kmm_is_simple_point(matrix: list[list[int]], x: int, y: int) -> bool:
    neighbors: set[tuple[int, int]] = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= ny < len(matrix) and 0 <= nx < len(matrix[0]) and matrix[ny][nx] > 0:
                neighbors.add((nx, ny))

    if len(neighbors) <= 1:
        return False

    remaining = set(neighbors)
    components = 0
    while remaining:
        components += 1
        if components > 1:
            return False
        stack = [remaining.pop()]
        while stack:
            cx, cy = stack.pop()
            connected = [
                point
                for point in list(remaining)
                if max(abs(point[0] - cx), abs(point[1] - cy)) == 1
            ]
            for point in connected:
                remaining.remove(point)
                stack.append(point)

    return True


def _kmm_is_deletable(matrix: list[list[int]], x: int, y: int) -> bool:
    if matrix[y][x] == 0:
        return False
    if _kmm_neighbor_count(matrix, x, y) <= 1:
        return False
    if not _kmm_touches_background(matrix, x, y):
        return False
    if _crossing_number(_ordered_neighbors_matrix(matrix, x, y)) != 1:
        return False
    weight = _kmm_weight_at(matrix, x, y)
    return weight in KMM_DELETION_WEIGHTS and _kmm_is_simple_point(matrix, x, y)


def deduplicate_minutiae(minutiae: Iterable[Minutia], radius: int = 6) -> list[Minutia]:
    kept: list[Minutia] = []
    for minutia in minutiae:
        if any(
            other.kind == minutia.kind and math.hypot(other.x - minutia.x, other.y - minutia.y) < radius
            for other in kept
        ):
            continue
        kept.append(minutia)
    return kept


def detect_minutiae(skeleton: BinaryImage, support_mask: BinaryImage, border_margin: int = 10) -> list[Minutia]:
    min_x, min_y, max_x, max_y = bounding_box(support_mask)
    min_x += border_margin
    min_y += border_margin
    max_x -= border_margin
    max_y -= border_margin

    candidates: list[Minutia] = []
    for y in range(1, skeleton.height - 1):
        for x in range(1, skeleton.width - 1):
            if not skeleton.pixels[y][x]:
                continue
            if x < min_x or x > max_x or y < min_y or y > max_y:
                continue
            neighbors = _ordered_neighbors(skeleton, x, y)
            cn = _crossing_number(neighbors)
            if cn == 1:
                candidates.append(Minutia(x, y, "ending"))
            elif cn == 3:
                candidates.append(Minutia(x, y, "bifurcation"))
    return deduplicate_minutiae(candidates)


def detect_minutiae_filtered(
    skeleton: BinaryImage,
    support_mask: BinaryImage,
    border_margin: int = 10,
    min_distance: int = 6,
) -> list[Minutia]:
    if min_distance < 1:
        raise ValueError("Minimalna odleglosc minucji musi byc >= 1.")
    candidates = detect_minutiae(skeleton, support_mask, border_margin=border_margin)
    return deduplicate_minutiae(candidates, radius=min_distance)


def connect_ridge_gaps(binary: BinaryImage, max_gap: int = 7, max_links_per_endpoint: int = 1) -> BinaryImage:
    if max_gap < 1:
        raise ValueError("Maksymalna przerwa musi byc >= 1.")
    if max_links_per_endpoint < 1:
        raise ValueError("Maksymalna liczba polaczen musi byc >= 1.")

    result = binary.copy()
    points = endpoints(result)
    if len(points) < 2:
        return result

    max_gap_sq = max_gap * max_gap
    link_budget = [0 for _ in points]
    pairs: list[tuple[int, int]] = []

    for idx, (x0, y0) in enumerate(points):
        if link_budget[idx] >= max_links_per_endpoint:
            continue
        best = -1
        best_dist = max_gap_sq + 1
        for jdx in range(idx + 1, len(points)):
            if link_budget[jdx] >= max_links_per_endpoint:
                continue
            x1, y1 = points[jdx]
            dx = x1 - x0
            dy = y1 - y0
            dist_sq = dx * dx + dy * dy
            if 0 < dist_sq <= max_gap_sq and dist_sq < best_dist:
                best_dist = dist_sq
                best = jdx
        if best >= 0:
            link_budget[idx] += 1
            link_budget[best] += 1
            pairs.append((idx, best))

    for first_idx, second_idx in pairs:
        x0, y0 = points[first_idx]
        x1, y1 = points[second_idx]
        for x, y in _bresenham_line(x0, y0, x1, y1):
            result.pixels[y][x] = 1
    return result


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Iterable[tuple[int, int]]:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx - dy
    x, y = x0, y0

    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        double_error = 2 * error
        if double_error > -dy:
            error -= dy
            x += step_x
        if double_error < dx:
            error += dx
            y += step_y


def component_count(binary: BinaryImage) -> int:
    return len(connected_components(binary))


def overlay_minutiae(
    base: GrayImage,
    skeleton: BinaryImage,
    minutiae: list[Minutia],
    show_endings: bool = True,
    show_bifurcations: bool = True,
) -> Image.Image:
    image = gray_to_pillow(base).convert("RGB")
    draw = ImageDraw.Draw(image)

    for y in range(skeleton.height):
        for x in range(skeleton.width):
            if skeleton.pixels[y][x]:
                draw.point((x, y), fill=(220, 0, 0))

    for minutia in minutiae:
        if minutia.kind == "ending" and not show_endings:
            continue
        if minutia.kind == "bifurcation" and not show_bifurcations:
            continue
        color = (0, 160, 0) if minutia.kind == "ending" else (0, 70, 220)
        draw.ellipse((minutia.x - 4, minutia.y - 4, minutia.x + 4, minutia.y + 4), outline=color, width=2)

    return image


def _build_result(
    label: str,
    base_gray: GrayImage,
    raw_skeleton: BinaryImage,
    support_mask: BinaryImage,
    bridge_iterations: int,
    spur_iterations: int,
    spur_length: int,
    bridge_gap: int,
    bridge_links_per_endpoint: int,
    min_distance: int,
) -> AlgorithmResult:
    repaired = bridge_gaps(raw_skeleton, iterations=bridge_iterations)
    repaired = connect_ridge_gaps(repaired, max_gap=bridge_gap, max_links_per_endpoint=bridge_links_per_endpoint)
    final = prune_short_spurs(repaired, max_length=spur_length, iterations=spur_iterations)
    minutiae = detect_minutiae_filtered(final, support_mask, min_distance=min_distance)
    endings = sum(1 for item in minutiae if item.kind == "ending")
    bifurcations = sum(1 for item in minutiae if item.kind == "bifurcation")
    summary = (
        f"{label}: piksele={final.count_foreground()} | skladowe={component_count(final)} | "
        f"zakonczenia={endings} | bifurkacje={bifurcations}"
    )
    overlay = overlay_minutiae(base_gray, final, minutiae)
    return AlgorithmResult(
        skeleton_raw=raw_skeleton,
        skeleton_final=final,
        minutiae=minutiae,
        overlay=overlay,
        summary=summary,
    )


def analyze_fingerprint(
    source: GrayImage,
    threshold_bias: int = 0,
    close_size: int = 3,
    open_size: int = 3,
    bridge_iterations: int = 1,
    spur_iterations: int = 1,
    spur_length: int = 4,
    bridge_gap: int = 7,
    bridge_links_per_endpoint: int = 1,
    binarization_method: str = "otsu",
    invert: bool = False,
    sauvola_window_size: int = 21,
    sauvola_k: float = 0.2,
    sauvola_dynamic_range: float = 128.0,
    mask_enabled: bool = False,
    mask_window_size: int = 17,
    mask_min_std: float = 12.0,
    mask_kernel_size: int = 9,
    minutiae_min_distance: int = 6,
) -> FingerprintAnalysis:
    enhanced = contrast_stretch(source)
    binary, threshold = binarize_fingerprint(
        enhanced,
        method=binarization_method,
        threshold=None,
        invert=invert,
        sauvola_window_size=sauvola_window_size,
        sauvola_k=sauvola_k,
        sauvola_dynamic_range=sauvola_dynamic_range,
    )
    if threshold_bias and binarization_method == "otsu":
        threshold = clamp(threshold + threshold_bias)
        binary = threshold_dark_ridges(enhanced, threshold)
        if invert:
            binary = BinaryImage(binary.width, binary.height, [[0 if v else 1 for v in row] for row in binary.pixels])
    if mask_enabled:
        mask = estimate_fingerprint_mask(
            enhanced,
            window_size=mask_window_size,
            min_std=mask_min_std,
            kernel_size=mask_kernel_size,
        )
        binary = binary_and(binary, mask)
    cleaned = remove_small_components(binary, min_area=10)
    if close_size > 1:
        cleaned = morph_close(cleaned, close_size)
    if open_size > 1:
        cleaned = morph_open(cleaned, open_size)

    morph_raw = morphological_skeleton(cleaned)
    k3m_raw = thin_k3m(cleaned)
    kmm_raw = thin_kmm(cleaned)
    morph_result = _build_result(
        label="Szkieletyzacja morfologiczna",
        base_gray=enhanced,
        raw_skeleton=morph_raw,
        support_mask=cleaned,
        bridge_iterations=bridge_iterations,
        spur_iterations=spur_iterations,
        spur_length=spur_length,
        bridge_gap=bridge_gap,
        bridge_links_per_endpoint=bridge_links_per_endpoint,
        min_distance=minutiae_min_distance,
    )
    k3m_result = _build_result(
        label="K3M",
        base_gray=enhanced,
        raw_skeleton=k3m_raw,
        support_mask=cleaned,
        bridge_iterations=bridge_iterations,
        spur_iterations=spur_iterations,
        spur_length=spur_length,
        bridge_gap=bridge_gap,
        bridge_links_per_endpoint=bridge_links_per_endpoint,
        min_distance=minutiae_min_distance,
    )
    kmm_result = _build_result(
        label="KMM",
        base_gray=enhanced,
        raw_skeleton=kmm_raw,
        support_mask=cleaned,
        bridge_iterations=bridge_iterations,
        spur_iterations=spur_iterations,
        spur_length=spur_length,
        bridge_gap=bridge_gap,
        bridge_links_per_endpoint=bridge_links_per_endpoint,
        min_distance=minutiae_min_distance,
    )

    comparison = (
        f"Prog={threshold} ({binarization_method}). Morfologia: {len(morph_result.minutiae)} minucji. "
        f"K3M: {len(k3m_result.minutiae)} minucji. KMM: {len(kmm_result.minutiae)} minucji."
    )
    return FingerprintAnalysis(
        source=source,
        enhanced=enhanced,
        binary=binary,
        cleaned=cleaned,
        threshold=threshold,
        morph=morph_result,
        k3m=k3m_result,
        kmm=kmm_result,
        comparison_summary=comparison,
    )

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Загрузка изображения."""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование из BGR в RGB

def calibrate_image(image):
    """Калибровка изображения (например, изменение яркости и контраста)."""
    # Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    calibrated_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return calibrated_image

def normalize_image(image):
    """Нормализация изображения."""
    # Приведение значений пикселей к диапазону [0, 1]
    normalized_image = image / 255.0
    return normalized_image

def denoise_image(image):
    """Удаление шумов из изображения с помощью медианного фильтра."""
    denoised_image = cv2.medianBlur(image, 5)  # Размер ядра 5
    return denoised_image

def display_images(original, calibrated, normalized, denoised):
    """Отображение изображений."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    
    plt.subplot(2, 2, 2)
    plt.title("Calibrated Image")
    plt.imshow(calibrated)
    
    plt.subplot(2, 2, 3)
    plt.title("Normalized Image")
    plt.imshow(normalized)
    
    plt.subplot(2, 2, 4)
    plt.title("Denoised Image")
    plt.imshow(denoised)
    
    plt.tight_layout()
    plt.show()

# Путь к изображению
image_path = 'path_to_your_image.jpg'  # Замените на путь к вашему изображению

# Выполнение обработки
original_image = load_image(image_path)
calibrated_image = calibrate_image(original_image)
normalized_image = normalize_image(calibrated_image)
denoised_image = denoise_image(calibrated_image)

# Отображение результатов
display_images(original_image, calibrated_image, normalized_image, denoised_image)

import os
import time
import re
import pytesseract
from concurrent.futures import ThreadPoolExecutor

USE_THREADS = int(os.getenv("USE_THREADS") or 1)
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS") or (os.cpu_count() // 2))
IMAGES_PER_CHUNK = int(os.getenv("IMAGES_PER_CHUNK") or 4)

def _ocr_image(image):
    return pytesseract.image_to_string(image, lang='ukr')

def _ocr_images_without_threads(images):
    ocr_start_time = time.time()
    print("OCR started")

    ocr_text = ""
    for img in images:
        text = _ocr_image(img)
        ocr_text += text + "\n"

    ocr_end_time = time.time()
    duration = ocr_end_time - ocr_start_time
    print(f"OCR finished. Time: {duration}s.")

    return ocr_text

def _ocr_images_with_threads(images):
    chunk_size = min(MAX_CHUNKS, len(images) // IMAGES_PER_CHUNK)

    ocr_start_time = time.time()
    print(f"OCR started with threading, len(images)={len(images)} cpu_count={os.cpu_count()} chunk_size={chunk_size}")

    ocr_text = [""] * len(images)  # Initialize a list to store results in order

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i + chunk_size]
            futures.append(executor.submit(lambda chunk, start_index=i: [(start_index + j, _ocr_image(img)) for j, img in enumerate(chunk)], chunk))

        for future in futures:
            results = future.result()
            for index, text in results:
                ocr_text[index] = text  # Place the result in the correct order

    ocr_text = "\n".join(ocr_text)

    ocr_end_time = time.time()
    duration = round(ocr_end_time - ocr_start_time, 2)
    print(f"OCR finished. Time: {duration}s.")

    return ocr_text

def ocr_images(images):
    ocr_images = _ocr_images_with_threads if USE_THREADS and len(images) > IMAGES_PER_CHUNK else _ocr_images_without_threads
    return ocr_images(images) if len(images) > 1 else _ocr_image(images[0])

# Remove hyphenation and merge lines
def remove_hyphenation(ocr_text):
    merged_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', ocr_text)
    merged_text = re.sub(r'(?<!\.\n)\n', ' ', merged_text)
    return merged_text
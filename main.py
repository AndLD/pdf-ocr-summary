from dotenv import load_dotenv
load_dotenv()
import os
import time
import uuid
import re
import json
import pytesseract
from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
from llama_index.core import VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext
from concurrent.futures import ThreadPoolExecutor

USE_THREADS=0

default_prompt = "Provide keywords and summary on a document. Use ukrainian language. Result should be in json format with fields 'keywords' and 'summary'."

app = FastAPI()

os.makedirs("debug", exist_ok=True)
os.makedirs("storage", exist_ok=True)

# Remove hyphenation and merge lines
def remove_hyphenation(ocr_text):
    merged_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', ocr_text)
    merged_text = re.sub(r'(?<!\.\n)\n', ' ', merged_text)
    return merged_text

# Query the model
def query_llama_index(index, query_text=default_prompt):
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query_text)

    try:
        return json.loads(response.response)
    except json.JSONDecodeError:
        return response.response

def ocr_image(image):
    return pytesseract.image_to_string(image, lang='ukr')

def ocr_images_without_threads(images):
    ocr_start_time = time.time()
    print("OCR started")

    # Perform OCR
    ocr_text = ""
    for img in images:
        text = ocr_image(img)
        ocr_text += text + "\n"

    ocr_end_time = time.time()
    duration = ocr_end_time - ocr_start_time
    print(f"OCR finished. Time: {duration}s.")

    return ocr_text

def ocr_images_with_threads(images):
    cpu_count = os.cpu_count()
    chunk_size = cpu_count if len(images) >= cpu_count else len(images)

    ocr_start_time = time.time()
    print(f"OCR started with threading, cpu_count={cpu_count} chunk_size={chunk_size}")

    ocr_text = [""] * len(images)  # Initialize a list to store results in order

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i + chunk_size]
            futures.append(executor.submit(lambda chunk, start_index=i: [(start_index + j, ocr_image(img)) for j, img in enumerate(chunk)], chunk))

        for future in futures:
            results = future.result()
            for index, text in results:
                ocr_text[index] = text  # Place the result in the correct order

    ocr_text = "\n".join(ocr_text)

    ocr_end_time = time.time()
    duration = ocr_end_time - ocr_start_time
    print(f"OCR finished. Time: {duration}s.")

    return ocr_text

def ocr_images(images):
    return ocr_images_with_threads(images) if USE_THREADS else ocr_images_without_threads(images)

# Route to upload PDF/PNG and process it
@app.post("/upload")
async def upload_file(file: UploadFile, ocr_only: int = 0):
    # Generate a unique ID for the file
    unique_id = str(uuid.uuid4())
    ocr_result_path = f"debug/ocr-result-{unique_id}.txt"
    document_path = f"debug/document-{unique_id}.txt"
    summary_path = f"debug/summary-{unique_id}.txt"

    # Handle file upload
    if file.filename.endswith(".pdf"):
        images = convert_from_bytes(file.file.read())
    elif file.filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file.file)
        images = [image]
    else:
        return {"message": "Unsupported file format. Please upload a PDF or PNG."}

    ocr_text = ocr_images(images) if len(images) > 1 else ocr_image(images[0])

    # Write the raw OCR result to a file
    with open(ocr_result_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    # Remove hyphenations and merge lines
    cleaned_text = remove_hyphenation(ocr_text)
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    if ocr_only:
        print('ocr_only is not 0, skipping AI part')

        return {"id": unique_id}

    # Vectorize document
    documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)

    index_store_path = f"storage/index-{unique_id}.json"
    index.storage_context.persist(persist_dir=index_store_path)

    # Query the model
    result = query_llama_index(index)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    return {"id": unique_id, "message": result}

# Route to query LlamaIndex by unique index ID
@app.get("/query/{id}")
async def query_index(id: str):
    index_store_path = f"storage/index-{id}.json"

    if not os.path.exists(index_store_path):
        return {"message": f"No index found for ID {id}"}

    # Load the stored index
    storage_context = StorageContext.from_defaults(persist_dir=index_store_path)
    index = load_index_from_storage(storage_context)

    result = query_llama_index(index)

    return {"id": id, "message": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

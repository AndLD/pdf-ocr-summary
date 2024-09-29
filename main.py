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

# Route to upload PDF/PNG and process it
@app.post("/upload")
async def upload_file(file: UploadFile):
    print(file.content_type)

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

    print(3)

    ocr_start_time = time.time()
    print("OCR started")

    # Perform OCR
    ocr_text = ""
    for img in images:
        text = pytesseract.image_to_string(img, lang='ukr')  # Assuming Ukrainian
        ocr_text += text + "\n"

    ocr_end_time = time.time()
    duration = ocr_end_time - ocr_start_time
    print(f"OCR finished. Time: {duration}s.")

    # Write the raw OCR result to a file
    with open(ocr_result_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    # Remove hyphenations and merge lines
    cleaned_text = remove_hyphenation(ocr_text)
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

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

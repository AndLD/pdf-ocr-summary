from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
from ocr import ocr_images, remove_hyphenation
from ai import create_index, load_index, query_index

app = FastAPI()

os.makedirs("debug", exist_ok=True)
os.makedirs("storage", exist_ok=True)

# Route to upload PDF/PNG and process it
@app.post("/upload")
async def upload_file(file: UploadFile, ocr_only: int = 0):
    # Generate a unique ID for the file
    unique_id = str(uuid.uuid4())
    ocr_result_path = f"debug/ocr-result-{unique_id}.txt"
    document_path = f"debug/document-{unique_id}.txt"
    index_store_path = f"storage/index-{unique_id}.json"
    summary_path = f"debug/summary-{unique_id}.txt"

    # Handle file upload
    if file.filename.endswith(".pdf"):
        images = convert_from_bytes(file.file.read())
    elif file.filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file.file)
        images = [image]
    else:
        return {"message": "Unsupported file format. Please upload a PDF or PNG."}

    ocr_text = ocr_images(images)

    # Write the raw OCR result to a file
    with open(ocr_result_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    # Remove hyphenations and merge lines
    cleaned_text = remove_hyphenation(ocr_text)
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    if ocr_only:
        print('ocr_only is set, skipping AI part')

        return {"id": unique_id}

    # Create index
    index = create_index(document_path, index_store_path)

    # Query the model
    result = query_index(index)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    return {"id": unique_id, "message": result}

# Route to query LlamaIndex by unique index ID
@app.get("/query/{id}")
async def query_summary(id: str):
    index_store_path = f"storage/index-{id}.json"

    if not os.path.exists(index_store_path):
        return {"message": f"No index found for ID {id}"}

    index = load_index(index_store_path)

    # Query the model
    result = query_index(index)

    return {"id": id, "message": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

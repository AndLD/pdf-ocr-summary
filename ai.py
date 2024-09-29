import json
from llama_index.core import VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext

default_prompt = "Provide keywords and summary on a document. Use ukrainian language. Result should be in json format with fields 'keywords' and 'summary'."

# Query the model
def query_index(index, query_text=default_prompt):
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query_text)

    try:
        return json.loads(response.response)
    except json.JSONDecodeError:
        return response.response

def create_index(document_path, index_store_path):
    # Vectorize document
    documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store index
    index.storage_context.persist(persist_dir=index_store_path)

    return index
    
# Load the stored index
def load_index(index_store_path):
    storage_context = StorageContext.from_defaults(persist_dir=index_store_path)
    index = load_index_from_storage(storage_context)

    return index
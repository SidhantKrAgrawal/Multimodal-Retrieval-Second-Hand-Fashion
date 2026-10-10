from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
from umap import UMAP
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
embedding_dir_dinov2 = BASE_DIR / "dataset2" / "img_embeddings_dinov2"
embedding_dir_clip = BASE_DIR / "dataset2" / "img_embeddings_clip"
embedding_dir_clip_title = BASE_DIR / "dataset2" / "title_embeddings_clip"
embedding_dir_clip_desc = BASE_DIR / "dataset2" / "desc_embeddings_clip"
embedding_dir_qwen_title = BASE_DIR / "dataset2" / "title_embeddings_qwen"
embedding_dir_qwen_desc = BASE_DIR / "dataset2" / "desc_embeddings_qwen"
labels_csv_dir = BASE_DIR / "dataset2" / "labels.csv"

def get_filtered_indices(filters: dict, labels_path: Path) -> list:
    df = pd.read_csv(labels_path)
    mask = pd.Series([True] * len(df))

    for key, values in filters.items():
        column_name = key.capitalize()
        if column_name in df.columns:
                column = df[column_name].astype(str).str.replace("^: ", "", regex=True).str.strip().str.lower()
            mask &= column.isin([v.lower() for v in values])

    return df[mask].index.tolist()



def get_top_matches(embedding_dir, filename, index_offset=0, filtered_indices=None):
    corrected_filename = str(int(filename) - index_offset)
    print(corrected_filename)
    embedding_path = embedding_dir / f"{corrected_filename}.pt"
    if not embedding_path.exists():
        return None, "Embedding not found."

    query_embedding = torch.load(embedding_path)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)

    all_files = sorted(list(embedding_dir.glob("*.pt")), key=lambda p: int(p.stem))
    all_image_names = [f.stem for f in all_files]

    if filtered_indices is not None:
        filtered_names = [str(i) for i in filtered_indices]
        filtered_files = [embedding_dir / f"{name}.pt" for name in filtered_names if (embedding_dir / f"{name}.pt").exists()]
        embeddings = []
        for f in filtered_files:
            emb = torch.load(f)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)
    else:
        filtered_files = all_files
        filtered_names = all_image_names
        embeddings = []
        for f in filtered_files:
            emb = torch.load(f)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)

    embedding_matrix = torch.cat(embeddings, dim=0)

    embedding_matrix = embedding_matrix / embedding_matrix.norm(p=2, dim=1, keepdim=True)
    query_embedding = query_embedding / query_embedding.norm(p=2, dim=1, keepdim=True)

    similarities = cosine_similarity(query_embedding.cpu().numpy(), embedding_matrix.cpu().numpy())[0]

    # exclude the query item itself from results
    if corrected_filename in filtered_names:
        query_index = filtered_names.index(corrected_filename)
        similarities[query_index] = -1

    top_k = min(10, len(similarities))
    nearest_indices = np.argsort(similarities)[-top_k:][::-1]

    actual_indices = [int(filtered_names[i]) for i in nearest_indices]
    top_matches = [f"{idx}.jpg" for idx in actual_indices]

    return top_matches, None



async def handle_upload(file: UploadFile = File(...), filters: str = Form(None), embedding_dir: Path = None, index_offset=0):
    try:
        filename = Path(file.filename).stem
        parsed_filters = json.loads(filters) if filters else None

        filtered_indices = None
        if parsed_filters:
            print("Parsed filters:", parsed_filters)
            filtered_indices = get_filtered_indices(parsed_filters, labels_csv_dir)
            print("Filtered indices:", len(filtered_indices))

        results, error = get_top_matches(embedding_dir, filename, index_offset, filtered_indices)
        if error:
            return JSONResponse(status_code=404, content={"error": error})
        return JSONResponse(content={"results": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload/dinov2")
async def upload_dinov2(file: UploadFile = File(...), filters: str = Form(None)):
    return await handle_upload(file=file, filters=filters, embedding_dir=embedding_dir_dinov2)

@app.post("/upload/clip")
async def upload_clip_image(file: UploadFile = File(...), filters: str = Form(None)):
    return await handle_upload(file=file, filters=filters, embedding_dir=embedding_dir_clip)

@app.post("/upload/clip-title")
async def upload_clip_title(file: UploadFile = File(...), filters: str = Form(None)):
    return await handle_upload(file=file, filters=filters, embedding_dir=embedding_dir_clip_title, index_offset=2)

@app.post("/upload/clip-desc")
async def upload_clip_desc(file: UploadFile = File(...), filters: str = Form(None)):
    return await handle_upload(file=file, filters=filters, embedding_dir=embedding_dir_clip_desc, index_offset=2)

@app.post("/upload/qwen-title")
async def upload_qwen_title(file: UploadFile = File(...), filters: str = Form(None)):
    return await handle_upload(file=file, filters=filters, embedding_dir=embedding_dir_qwen_title, index_offset=2)

@app.post("/upload/qwen-desc")
async def upload_qwen_desc(file: UploadFile = File(...), filters: str = Form(None)):
    return await handle_upload(file=file, filters=filters, embedding_dir=embedding_dir_qwen_desc, index_offset=2)

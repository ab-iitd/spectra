import os
import io
import logging
from typing import Optional
import ast
import base64

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import asyncio

from transformers import (
    BlipProcessor, BlipForConditionalGeneration
)

TRAIN_PARQUET_PATH = "/data/train.parquet"
TEST_PARQUET_PATH = "/data/test.parquet"
DEVICE = os.getenv("CAPTION_DEVICE", "cpu")
BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
CAPTION_TIMEOUT = 45

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Image_tools_service")
app = FastAPI()

# Data
logger.info("Loading parquet datasets...")
train_df = pd.read_parquet(TRAIN_PARQUET_PATH) if os.path.exists(TRAIN_PARQUET_PATH) else None
test_df = pd.read_parquet(TEST_PARQUET_PATH) if os.path.exists(TEST_PARQUET_PATH) else None

# Loading BLIP Model
logger.info(f"Loading BLIP model {BLIP_MODEL_NAME} on {DEVICE}")
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)
blip_model.eval()
logger.info("BLIP model loaded successfully.")

class ImageURLRequest(BaseModel):
    image_url: str

# Helpers
def _safe_get_image_bytes_from_row(row: pd.Series) -> Optional[bytes]:
    try:
        images = row.get("images", None)
        if not images or len(images) == 0:
            return None
        first = images[0]
        if isinstance(first, dict):
            for k in ("bytes", "data", "b", "img_bytes", "content"):
                if k in first:
                    val = first[k]
                    if isinstance(val, (bytes, bytearray, memoryview)):
                        return bytes(val)
                    if isinstance(val, str):
                        try:
                            return base64.b64decode(val)
                        except Exception:
                            pass
            for val in first.values():
                if isinstance(val, (bytes, bytearray, memoryview)):
                    return bytes(val)
        elif isinstance(first, (bytes, bytearray, memoryview)):
            return bytes(first)
    except Exception:
        logger.exception("Error extracting image bytes from row")
    return None

def search_datasets_for_image(image_id: str) -> Optional[bytes]:
    for df_name, df in [("train", train_df), ("test", test_df)]:
        if df is None or "image_link" not in df.columns:
            continue

        for idx, row in df.iterrows():
            links = row["image_link"]
            if isinstance(links, str):
                links = [links]

            flat_links = []
            for x in links:
                if isinstance(x, (list, tuple)):
                    flat_links.extend([str(i).strip() for i in x])
                elif x is not None:
                    flat_links.append(str(x).strip())

            if image_id in flat_links:
                logger.info(f"Found image '{image_id}' in {df_name} dataset at row {idx}")
                return _safe_get_image_bytes_from_row(row)

    logger.warning(f"Image '{image_id}' not found in any dataset")
    return None

def pil_from_bytes(blob: bytes) -> Image.Image:
    return Image.open(io.BytesIO(blob)).convert("RGB")

async def run_caption_in_thread(image: Image.Image) -> str:
    loop = asyncio.get_running_loop()
    def _inference(img: Image.Image) -> str:
        inputs = blip_processor(img, return_tensors="pt")
        input_tensors = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = blip_model.generate(**input_tensors)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    return await loop.run_in_executor(None, _inference, image)

# ---------- Endpoints ----------
# Image Captioning tool
@app.post("/captioning_tool")
async def captioning_tool(req: ImageURLRequest):
    if not blip_model or not blip_processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    image_id = req.image_url.strip()
    if not image_id:
        raise HTTPException(status_code=400, detail="image_url must be a non-empty string")
    async def process_caption():
        bytes_blob = search_datasets_for_image(image_id)
        if not bytes_blob:
            raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found in datasets")
        img = pil_from_bytes(bytes_blob)
        return await run_caption_in_thread(img)
    try:
        caption = await asyncio.wait_for(process_caption(), timeout=CAPTION_TIMEOUT)
        return {"success": True, "message": caption}
    except asyncio.TimeoutError:
        return {"success": False, "message": f"Captioning timed out after {CAPTION_TIMEOUT} seconds"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error during captioning")
        return {"success": False, "message": f"Captioning failed: {str(e)}"}

@app.get("/health")
async def health_check():
    model_ready = bool(blip_model and blip_processor)
    return {"status": "healthy" if model_ready else "degraded", "model_loaded": model_ready}

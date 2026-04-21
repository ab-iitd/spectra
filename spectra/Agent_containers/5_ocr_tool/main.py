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
import pytesseract

TRAIN_PARQUET_PATH = "/data/train.parquet"
TEST_PARQUET_PATH = "/data/test.parquet"
DEVICE = os.getenv("CAPTION_DEVICE", "cpu")
OCR_TIMEOUT = 45

# Pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Image_tools_service")
app = FastAPI()

#  Data
logger.info("Loading parquet datasets...")
train_df = pd.read_parquet(TRAIN_PARQUET_PATH) if os.path.exists(TRAIN_PARQUET_PATH) else None
test_df = pd.read_parquet(TEST_PARQUET_PATH) if os.path.exists(TEST_PARQUET_PATH) else None

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

async def run_ocr_in_thread(image: Image.Image):
    loop = asyncio.get_running_loop()
    def _ocr(img: Image.Image):
        text = pytesseract.image_to_string(img)
        return text.strip()
    return await loop.run_in_executor(None, _ocr, image)

# Endpoints
# OCR tool
@app.post("/ocr_tool")
async def ocr_tool(req: ImageURLRequest):
    image_id = req.image_url.strip()
    if not image_id:
        raise HTTPException(status_code=400, detail="image_url must be a non-empty string")
    async def process_ocr():
        bytes_blob = search_datasets_for_image(image_id)
        if not bytes_blob:
            raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found in datasets")
        img = pil_from_bytes(bytes_blob)
        return await run_ocr_in_thread(img)
    try:
        text = await asyncio.wait_for(process_ocr(), timeout=OCR_TIMEOUT)
        if not text:
            return {"success": False, "message": "No text detected in image."}
        return {"success": True, "message": text}
    except asyncio.TimeoutError:
        return {"success": False, "message": f"OCR timed out after {OCR_TIMEOUT} seconds"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error during OCR")
        return {"success": False, "message": f"OCR failed: {str(e)}"}

@app.get("/health")
async def health_check():
    model_ready = bool(blip_model and blip_processor and detr_model and detr_processor)
    return {"status": "healthy" if model_ready else "degraded", "model_loaded": model_ready}

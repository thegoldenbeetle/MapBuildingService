#!/usr/bin/env python3
import io
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image


from .model import DetectionModel
from .utils import draw_lines, get_mask_image, save_image

DATA_PATH = Path("/tmp/map_builder_service_data")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")
model = DetectionModel()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("page.html", {"request": request})


@app.get("/image/{image_name}")
async def get_image(image_name: str):
    return FileResponse(DATA_PATH / image_name)


@app.post("/api/detect")
async def detect(request: Request, img_file: UploadFile = Form(...)):
    allowedFiles = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/tiff",
        "image/bmp",
        "video/webm",
    }

    if img_file.content_type in allowedFiles:
        image_data = await img_file.read()
        image = Image.open(io.BytesIO(image_data))

        detect_result = model.detect_line(image)

        mask_img = get_mask_image(image, detect_result.mask)
        lines_img = draw_lines(image, detect_result.lines_2d)

        mask_img_path = save_image(mask_img)
        lines_img_path = save_image(lines_img)

        return {
            "mask": detect_result.mask.tolist(),
            "mask_image": mask_img_path,
            "lines_image": lines_img_path,
            "lines2d": [item.tolist() for item in detect_result.lines_2d],
        }
    else:
        return {"error": f"content_type {img_file.content_type} is not allowed."}

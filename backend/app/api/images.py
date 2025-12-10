"""Image generation and management API endpoints"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.services.image_service import get_image_service
from app.utils.logger import get_logger

logger = get_logger()
router = APIRouter()


class ImageGenerateRequest(BaseModel):
    """Request to generate an image"""
    prompt: str
    session_id: Optional[str] = None


class ImageGenerateResponse(BaseModel):
    """Response for image generation"""
    status: str
    image_url: str
    temp_id: str


class ImageSaveRequest(BaseModel):
    """Request to save a temporary image"""
    temp_id: str
    custom_filename: Optional[str] = None


class ImageSaveResponse(BaseModel):
    """Response for saving an image"""
    status: str
    filename: str
    path: str


@router.post("/images/generate", response_model=ImageGenerateResponse)
async def generate_image(request: ImageGenerateRequest) -> ImageGenerateResponse:
    """Generate an image from a prompt"""
    logger.info(f"Generating image: {request.prompt[:50]}...")

    image_service = get_image_service()

    try:
        result = await image_service.generate_image(request.prompt)

        return ImageGenerateResponse(
            status="success",
            image_url=result["image_url"],
            temp_id=result["temp_id"]
        )
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/save", response_model=ImageSaveResponse)
async def save_image(request: ImageSaveRequest) -> ImageSaveResponse:
    """Save a temporary image permanently"""
    logger.info(f"Saving image: {request.temp_id}")

    image_service = get_image_service()

    try:
        result = await image_service.save_image(request.temp_id, request.custom_filename)

        return ImageSaveResponse(
            status="success",
            filename=result["filename"],
            path=result["path"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images")
async def list_saved_images() -> Dict[str, List[Dict[str, Any]]]:
    """List all saved images"""
    image_service = get_image_service()
    images = await image_service.list_saved_images()

    return {"images": images}


@router.get("/images/session")
async def list_session_images() -> Dict[str, List[Dict[str, Any]]]:
    """List images from current session (saved + temporary)"""
    image_service = get_image_service()
    return await image_service.list_session_images()


@router.get("/images/temp/{temp_id}.jpg")
async def get_temp_image(temp_id: str):
    """Serve a temporary image"""
    image_service = get_image_service()

    if temp_id not in image_service.temp_images:
        raise HTTPException(status_code=404, detail="Temporary image not found")

    file_path = image_service.temp_images[temp_id]["path"]

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(file_path, media_type="image/jpeg")


@router.get("/images/saved/{filename}")
async def get_saved_image(filename: str):
    """Serve a saved image"""
    image_service = get_image_service()
    file_path = Path(image_service.save_directory) / filename

    # Security check
    if not str(file_path.resolve()).startswith(str(Path(image_service.save_directory).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path, media_type="image/jpeg")

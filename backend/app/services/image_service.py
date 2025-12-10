"""Image generation and management service"""

import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib

import aiofiles

from app.services.llm_service import create_llm_provider
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger()


class ImageService:
    """Service for generating and managing images"""

    def __init__(self):
        config = get_config()
        self.save_directory = config.get('images.save_directory', 'data/images')
        self.temp_directory = f"{self.save_directory}/temp"
        self.provider_name = config.get('images.provider', 'openai')

        # Ensure directories exist
        Path(self.save_directory).mkdir(parents=True, exist_ok=True)
        Path(self.temp_directory).mkdir(parents=True, exist_ok=True)

        # Track temporary images
        self.temp_images: Dict[str, Dict[str, Any]] = {}

    async def generate_image(self, prompt: str) -> Dict[str, str]:
        """Generate an image from a prompt"""
        logger.info(f"Generating image with prompt: {prompt[:50]}...")

        # Create LLM provider
        provider = create_llm_provider(self.provider_name)

        # Generate image
        image_bytes = await provider.generate_image(prompt)

        # Save to temporary location
        temp_id = str(uuid.uuid4())
        temp_path = f"{self.temp_directory}/{temp_id}.jpg"

        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(image_bytes)

        # Track temporary image
        self.temp_images[temp_id] = {
            "path": temp_path,
            "prompt": prompt,
            "generated_at": datetime.utcnow()
        }

        logger.info(f"Image generated and saved to {temp_path}")

        return {
            "temp_id": temp_id,
            "image_url": f"/api/images/temp/{temp_id}.jpg"
        }

    async def save_image(self, temp_id: str, custom_filename: Optional[str] = None) -> Dict[str, str]:
        """Save a temporary image permanently"""
        if temp_id not in self.temp_images:
            raise ValueError(f"Temporary image not found: {temp_id}")

        temp_info = self.temp_images[temp_id]
        temp_path = temp_info["path"]

        # Generate permanent filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(temp_id.encode()).hexdigest()[:8]

        if custom_filename:
            # Sanitize custom filename
            safe_filename = "".join(c for c in custom_filename if c.isalnum() or c in (' ', '-', '_'))
            safe_filename = safe_filename.replace(' ', '_')
            filename = f"{timestamp}_{hash_part}_{safe_filename}.jpg"
        else:
            filename = f"{timestamp}_{hash_part}.jpg"

        permanent_path = f"{self.save_directory}/{filename}"

        # Move file
        import shutil
        shutil.move(temp_path, permanent_path)

        # Remove from temp tracking
        del self.temp_images[temp_id]

        logger.info(f"Image saved permanently: {filename}")

        return {
            "filename": filename,
            "path": permanent_path
        }

    async def list_saved_images(self) -> List[Dict[str, Any]]:
        """List all saved images"""
        images = []
        save_dir = Path(self.save_directory)

        for file_path in save_dir.glob("*.jpg"):
            if file_path.parent.name == "temp":
                continue

            stat = file_path.stat()
            images.append({
                "filename": file_path.name,
                "url": f"/api/images/saved/{file_path.name}",
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "size_bytes": stat.st_size
            })

        # Sort by creation time, newest first
        images.sort(key=lambda x: x["created_at"], reverse=True)

        return images

    async def list_session_images(self) -> Dict[str, List[Dict[str, Any]]]:
        """List both saved and temporary images"""
        saved = await self.list_saved_images()

        temporary = []
        for temp_id, info in self.temp_images.items():
            temporary.append({
                "temp_id": temp_id,
                "url": f"/api/images/temp/{temp_id}.jpg",
                "prompt": info["prompt"],
                "generated_at": info["generated_at"]
            })

        return {
            "saved": saved,
            "temporary": temporary
        }

    async def cleanup_temp_images(self):
        """Clean up old temporary images"""
        cutoff_time = datetime.utcnow().timestamp() - (24 * 3600)  # 24 hours

        to_remove = []
        for temp_id, info in self.temp_images.items():
            if info["generated_at"].timestamp() < cutoff_time:
                # Delete file
                try:
                    os.remove(info["path"])
                    to_remove.append(temp_id)
                    logger.info(f"Cleaned up old temporary image: {temp_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp image {temp_id}: {e}")

        # Remove from tracking
        for temp_id in to_remove:
            del self.temp_images[temp_id]


# Global instance
_image_service = None


def get_image_service() -> ImageService:
    """Get the global image service instance"""
    global _image_service
    if _image_service is None:
        _image_service = ImageService()
    return _image_service

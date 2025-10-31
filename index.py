# main.py

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# --- Configuration ---
# The base URL for the external Creart AI API
CREART_AI_BASE_URL = "https://api.creartai.com/api/v1"

# --- Initialize FastAPI App ---
# This is the main object that runs your entire API
app = FastAPI(
    title="My Custom Image Generation API",
    description="A Python wrapper for the Creart AI text-to-image and image-to-image APIs.",
    version="1.0.0"
)

# --- Pydantic Models for Request Data ---
# These models define the structure of the data your API expects.
# FastAPI uses them to validate incoming requests automatically.

class Text2ImageRequest(BaseModel):
    """Defines the expected data for a text-to-image request."""
    prompt: str = Field(..., example="a beautiful landscape, cinematic lighting, 8k")
    negative_prompt: str = ""
    aspect_ratio: str = Field(default="1x1", example="4x5")
    guidance_scale: float = Field(default=9.5, example=8.0)
    seed: Optional[int] = None
    # Add any other parameters from the original API you want to support
    
    class Config:
        # This allows FastAPI to generate a sample UI in the docs
        schema_extra = {
            "example": {
                "prompt": "a majestic lion in the savannah, national geographic photo",
                "negative_prompt": "cartoon, drawing, blurry",
                "aspect_ratio": "16x9",
                "guidance_scale": 7.5
            }
        }

class Image2ImageRequest(Text2ImageRequest):
    """
    Defines the expected data for an image-to-image request.
    It inherits all fields from Text2ImageRequest and adds one more.
    """
    input_image_base64: str = Field(..., description="The source image encoded in Base64 format.")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "make the lion wear a crown, fantasy art",
                "input_image_base64": "/9j/4AAQSkZJRgABAQ... (very long string)",
                "negative_prompt": "blurry, low quality",
                "aspect_ratio": "1x1",
                "guidance_scale": 9.0
            }
        }


# --- API Endpoints ---

@app.post("/api/text-to-image")
async def generate_text_to_image(request_data: Text2ImageRequest):
    """
    Endpoint to generate an image from a text prompt.
    It takes the prompt and other parameters, then calls the external API.
    """
    # The endpoint path on the Creart AI server
    api_url = f"{CREART_AI_BASE_URL}/text2image"
    
    # Prepare the form data payload for the external API
    payload = {
        "prompt": request_data.prompt,
        "input_image_type": "text2image",
        "input_image_base64": "", # Required but empty for text2image
        "negative_prompt": request_data.negative_prompt,
        "aspect_ratio": request_data.aspect_ratio,
        "guidance_scale": request_data.guidance_scale,
        "seed": request_data.seed,
    }
    
    # Use httpx to make an asynchronous POST request
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"Forwarding request to: {api_url}")
            response = await client.post(api_url, data=payload)
            
            # Raise an exception if the external API returned an error (e.g., 4xx or 5xx)
            response.raise_for_status()
            
            # Return the successful response from the external API directly to the user
            return response.json()
            
        except httpx.HTTPStatusError as e:
            # If the external API fails, forward its error status and message
            raise HTTPException(
                status_code=e.response.status_code, 
                detail=f"Error from external API: {e.response.text}"
            )
        except Exception as e:
            # Handle other errors like network issues
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


@app.post("/api/image-to-image")
async def generate_image_to_image(request_data: Image2ImageRequest):
    """
    Endpoint to modify an image based on a text prompt.
    It takes a base64 encoded image, a prompt, and other parameters.
    """
    api_url = f"{CREART_AI_BASE_URL}/image2image"
    
    # Prepare the form data payload for the external API
    payload = {
        "prompt": request_data.prompt,
        "input_image_type": "image2image",
        "input_image_base64": request_data.input_image_base64,
        "negative_prompt": request_data.negative_prompt,
        "aspect_ratio": request_data.aspect_ratio,
        "guidance_scale": request_data.guidance_scale,
        "seed": request_data.seed,
    }
    
    # Use httpx to make an asynchronous POST request
    async with httpx.AsyncClient(timeout=120.0) as client: # Longer timeout for image processing
        try:
            print(f"Forwarding request to: {api_url}")
            response = await client.post(api_url, data=payload)
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, 
                detail=f"Error from external API: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


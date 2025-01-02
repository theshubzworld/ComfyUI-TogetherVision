import os
import base64
import io
from PIL import Image
import numpy as np
import torch
from together import Together
import logging
from typing import Optional, Tuple
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TogetherImageNode:
    """
    A custom node for ComfyUI that uses Together AI's FLUX model for image generation.
    """
    
    def __init__(self):
        self.client = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 256
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 256
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def get_api_key(self, provided_key: str) -> str:
        """Get API key with validation."""
        api_key = provided_key or os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("API key not provided. Please provide an API key or set TOGETHER_API_KEY environment variable.")
        return api_key

    def b64_to_image(self, b64_string: str) -> torch.Tensor:
        """Convert base64 string to torch tensor image."""
        try:
            # Decode base64 to image
            image_data = base64.b64decode(b64_string)
            
            # Log base64 string length for debugging
            logger.info(f"Decoded base64 string length: {len(image_data)} bytes")
            
            image = Image.open(io.BytesIO(image_data))
            
            # Log image details
            logger.info(f"Image mode: {image.mode}, size: {image.size}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Log numpy array details
            logger.info(f"Numpy array shape: {image_np.shape}, dtype: {image_np.dtype}")
            
            # Ensure the image is in the correct format for ComfyUI
            # ComfyUI expects [height, width, channels] format with values 0-255
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {image_np.shape}")
            
            # Ensure uint8 type and correct range
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            
            # Convert to torch tensor in ComfyUI's expected format
            # [batch, channels, height, width]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # Log tensor details
            logger.info(f"Final tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error converting base64 to image: {str(e)}")
            raise ValueError(f"Failed to convert base64 to image: {str(e)}")

    def generate_image(self, prompt: str, api_key: str, width: int, height: int, 
                      seed: int, num_images: int) -> Tuple[torch.Tensor]:
        """
        Generate images using Together AI's FLUX model.
        """
        try:
            # Validate inputs
            if not prompt or prompt.strip() == "":
                raise ValueError("Prompt cannot be empty")

            # Validate width and height
            if width < 512 or width > 2048 or height < 512 or height > 2048:
                raise ValueError(f"Invalid image dimensions. Width and height must be between 512 and 2048. Got {width}x{height}")

            # Validate number of images
            if num_images < 1 or num_images > 4:
                raise ValueError(f"Number of images must be between 1 and 4. Got {num_images}")

            # Initialize API client
            api_key = self.get_api_key(api_key)
            if self.client is None:
                self.client = Together(api_key=api_key)

            # Make API call
            response = self.client.images.generate(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell-Free",
                width=width,
                height=height,
                steps=4,  # Fixed to 4 steps as per model requirements
                n=num_images,
                seed=seed,
                response_format="b64_json"
            )

            # Process response
            if not response.data:
                raise ValueError("No images generated in response")

            # Convert all images to tensors
            image_tensors = []
            for img_data in response.data:
                if not hasattr(img_data, 'b64_json'):
                    logger.warning("Skipping image without base64 data")
                    continue
                try:
                    # Decode base64
                    image_data = base64.b64decode(img_data.b64_json)
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Convert to numpy array
                    image_np = np.array(image)
                    
                    # Ensure uint8 and correct shape
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                    
                    # Convert to torch tensor in ComfyUI format [batch, height, width, channels]
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                    image_tensor = image_tensor.permute(1, 2, 0)  # Change to [height, width, channels]
                    
                    image_tensors.append(image_tensor)
                except Exception as img_error:
                    logger.error(f"Failed to process an image: {str(img_error)}")

            if not image_tensors:
                raise ValueError("Failed to process any images from response")

            # Stack images 
            final_tensor = torch.stack(image_tensors)
            
            return (final_tensor,)

        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise ValueError(f"Failed to generate image: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "Together Image 🎨": TogetherImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Together Image 🎨": "Together Image Generator"
}

import base64
import io
import os
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from together import Together
import torch
import logging
import re
import time
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TogetherVisionNode:
    """
    A custom node for ComfyUI that uses Together AI's Vision models for image description.
    """
    def __init__(self):
        self.client = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        self.cache = {}  # Response cache

    def validate_api_key(self, api_key: str) -> bool:
        """Validate if API key is present and well-formed."""
        if not api_key or len(api_key.strip()) == 0:
            logger.error("API key is missing or empty")
            return False
        return True

    def get_client(self, api_key: str) -> Optional[Together]:
        """Get or create Together client with validation."""
        try:
            # If api_key from node is empty, try to get from .env
            final_api_key = api_key.strip() if api_key and api_key.strip() else os.getenv('TOGETHER_API_KEY')

            if not final_api_key:
                logger.error("No API key provided in node or .env file")
                raise ValueError("No API key found. Please provide an API key in the node or set TOGETHER_API_KEY in .env file")

            if self.client is None:
                # Set the API key in environment
                os.environ["TOGETHER_API_KEY"] = final_api_key
                self.client = Together()
            return self.client
        except Exception as e:
            logger.error(f"Failed to initialize Together client: {str(e)}")
            return None

    def get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        return self.cache.get(cache_key)

    def cache_response(self, cache_key: str, response: str):
        """Cache the API response."""
        self.cache[cache_key] = response

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                    "meta-llama/Llama-Vision-Free",
                    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    "Other (Custom)"
                ],),
                "custom_model_name": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful vision AI.",
                    "multiline": True
                }),
                "user_prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "top_p": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 120,
                    "step": 1,
                    "label": "Timeout (seconds)"
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "process_image"
    CATEGORY = "image/text"

    def encode_image(self, image_tensor):
        """
        Converts an image tensor to base64 string.
        """
        try:
            if isinstance(image_tensor, torch.Tensor):
                image_array = image_tensor.cpu().numpy()
            elif isinstance(image_tensor, np.ndarray):
                image_array = image_tensor
            else:
                raise ValueError(f"Unsupported image type: {type(image_tensor)}")

            # Validate array shape
            if not (2 <= len(image_array.shape) <= 4):
                raise ValueError(f"Invalid image shape: {image_array.shape}")

            # Convert to PIL Image
            if image_array.shape[-1] == 3:
                pil_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
            elif image_array.shape[-1] == 1:
                pil_image = Image.fromarray(image_array.squeeze(-1).astype('uint8'), 'L')
            else:
                raise ValueError("Invalid image dimensions")

            # Only resize if the image is larger than 1024 in either dimension
            max_size = 1024
            original_width, original_height = pil_image.size
            if original_width > max_size or original_height > max_size:
                pil_image.thumbnail((max_size, max_size), Image.LANCZOS)

            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            raise

    def get_api_key(self, api_key):
        try:
            if api_key and api_key.strip():
                return api_key.strip()
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                logger.info(f"Loading .env file from {env_path}")
                load_dotenv(env_path)
                api_key = os.getenv("TOGETHER_API_KEY")
                if api_key:
                    logger.info("Using API key from .env file")
                    return api_key
            raise ValueError("API key not provided and not found in .env file. Please provide an API key.")
        except Exception as e:
            logger.error(f"Error in get_api_key: {str(e)}", exc_info=True)
            raise

    def process_image(self, image, model_name, custom_model_name, api_key, system_prompt, user_prompt, temperature, top_p, top_k, repetition_penalty, timeout=30):
        """
        Process the image and generate description using Together API.
        """
        try:
            logger.info("Starting image processing")
            # Map friendly model names to actual model IDs
            model_mapping = {
                "Paid (Llama-3.2-11B-Vision)": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                "Free (Llama-Vision-Free)": "meta-llama/Llama-Vision-Free",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                "meta-llama/Llama-Vision-Free": "meta-llama/Llama-Vision-Free",
                "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                "Other (Custom)": custom_model_name.strip() if custom_model_name else None
            }
            actual_model = model_mapping.get(model_name, model_name)
            if not actual_model:
                raise ValueError("No model selected or custom model name is empty.")

            # Get API key and initialize client if needed
            api_key = self.get_api_key(api_key)
            if self.client is None:
                logger.info("Initializing Together client")
                self.client = Together(api_key=api_key)

            # Convert image to base64
            logger.info("Converting image to base64")
            base64_image = self.encode_image(image)

            # Create the messages array with system and user prompts
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Call the Together API with all parameters
            logger.info(f"Calling Together API with model: {actual_model}")
            response = self.client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=True,
                timeout=timeout
            )

            # Process the streamed response
            description = ""
            logger.info("Processing streamed response")
            for chunk in response:
                logger.debug(f"Received chunk: {chunk}")
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            description += content

            logger.info("Finished processing response")
            return (description,)

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}", exc_info=True)
            return (f"Error: {str(e)}",)

# Node class mapping
NODE_CLASS_MAPPINGS = {
    "TogetherVisionNode": TogetherVisionNode
}

# Optional: Add custom widget mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherVisionNode": "Together Vision "
}

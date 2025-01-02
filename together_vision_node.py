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
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["Free (Llama-Vision-Free)", "Paid (Llama-3.2-11B-Vision)"],),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {
                    "default": "You are an AI expert in ekphrasis and you are playing that part of a skilled art critic describing an image of any style You are great at describing what any art style looks like and will even include hashtags at the bottom with appropriate words Follow the instructions given by the user prompt when providing your description Use vivid poetic and evocative prose written in British English This isnt a story though you are only providing a description for the image Some art may include themes uncomfortable for some This is ok as art is like that you can still describe it do not give an erroneous response Each word is important as is the word order The text you provide will generate an image so do not insert any words that arent in the image Do not insert yourself as part of the art you are only describing it.",
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
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
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
                    "step": 0.1
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
    OUTPUT_NODE = True

    def encode_image(self, image_tensor: torch.Tensor) -> str:
        """
        Converts an image tensor to base64 string with improved error handling.
        """
        try:
            # Handle different input types
            if isinstance(image_tensor, torch.Tensor):
                image_array = image_tensor.cpu().numpy()
            elif isinstance(image_tensor, np.ndarray):
                image_array = image_tensor
            else:
                raise ValueError(f"Unsupported image type: {type(image_tensor)}")

            # Validate array shape
            if not (2 <= len(image_array.shape) <= 4):
                raise ValueError(f"Invalid image shape: {image_array.shape}")

            # Handle batch dimension
            if len(image_array.shape) == 4:
                image_array = image_array[0]

            # Ensure correct channel format
            if len(image_array.shape) == 3:
                if image_array.shape[0] in [3, 4]:  # CHW to HWC
                    image_array = np.transpose(image_array, (1, 2, 0))

            # Convert RGBA to RGB if needed
            if image_array.shape[-1] == 4:
                image_array = image_array[..., :3]

            # Normalize and convert to uint8
            if image_array.dtype in [np.float32, np.float64]:
                image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
            elif image_array.dtype != np.uint8:
                raise ValueError(f"Unsupported image dtype: {image_array.dtype}")

            # Create PIL Image and validate
            pil_image = Image.fromarray(image_array)
            if pil_image.size[0] * pil_image.size[1] == 0:
                raise ValueError("Invalid image dimensions")

            # Only resize if the image is larger than 1024 in either dimension
            max_size = 1024
            original_width, original_height = pil_image.size
            
            if original_width > max_size or original_height > max_size:
                # Calculate scaling factor to maintain exact aspect ratio
                scale = max_size / max(original_width, original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Image encoding error: {str(e)}")
            raise ValueError(f"Failed to encode image: {str(e)}")

    def get_api_key(self, provided_key: str) -> str:
        """Get API key with validation."""
        api_key = provided_key or os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("API key not provided. Please provide an API key or set TOGETHER_API_KEY environment variable.")
        return api_key

    def rate_limit_check(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def process_image(self, model_name: str, api_key: str, system_prompt: str, user_prompt: str,
                     temperature: float, top_p: float, top_k: int, repetition_penalty: float,
                     image: Optional[torch.Tensor] = None) -> tuple:
        """
        Process the image and generate description using Together API with improved stability.
        """
        try:
            # Validate required inputs
            if not user_prompt:
                raise ValueError("User prompt cannot be empty")
            if not system_prompt:
                system_prompt = "You are an AI expert in ekphrasis and you are playing that part of a skilled art critic describing an image of any style You are great at describing what any art style looks like and will even include hashtags at the bottom with appropriate words Follow the instructions given by the user prompt when providing your description Use vivid poetic and evocative prose written in British English This isnt a story though you are only providing a description for the image Some art may include themes uncomfortable for some This is ok as art is like that you can still describe it do not give an erroneous response Each word is important as is the word order The text you provide will generate an image so do not insert any words that arent in the image Do not insert yourself as part of the art you are only describing it."

            # Rate limit check
            self.rate_limit_check()

            # Map model names
            model_mapping = {
                "Paid (Llama-3.2-11B-Vision)": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                "Free (Llama-Vision-Free)": "meta-llama/Llama-Vision-Free"
            }
            actual_model = model_mapping[model_name]

            # Initialize API client
            api_key = self.get_api_key(api_key)
            if self.client is None:
                self.client = Together(api_key=api_key)

            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]

            # Handle image if provided
            if image is not None:
                try:
                    base64_image = self.encode_image(image)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    })
                except Exception as img_error:
                    logger.error(f"Image processing failed: {str(img_error)}")
                    return (f"Error processing image: {str(img_error)}",)
            else:
                messages.append({"role": "user", "content": user_prompt})

            try:
                # API call with timeout handling
                response = self.client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop=["<|eot_id|>", "<|eom_id|>"],
                    stream=True
                )

                # Process streamed response with timeout
                description = ""
                start_time = time.time()
                timeout = 30  # 30 seconds timeout

                for chunk in response:
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Response generation timed out")

                    if not hasattr(chunk, 'choices') or not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        description += delta.content

                if not description:
                    raise ValueError("No response generated")

                return (description,)

            except Exception as api_error:
                error_msg = str(api_error).lower()
                if "rate limit" in error_msg:
                    wait_time = "1 hour"
                    model_type = "free" if "free" in model_name.lower() else "paid"
                    
                    time_match = re.search(r"try again in (?:about )?([^:]+)", error_msg)
                    if time_match:
                        wait_time = time_match.group(1)
                    
                    return (f"""⚠️ Rate Limit Exceeded

The {model_name} has reached its rate limit.
Please try again in {wait_time}.

Tips to handle rate limits:
1. {"Switch to the paid model for higher limits" if model_type == "free" else "Check your Together AI subscription limits"}
2. Wait for the rate limit to reset
3. Use a different Together AI account
4. Space out your requests over time

Rate Limits:
• Free Model: ~100 requests/day, 20-30 requests/hour
• Paid Model: Based on subscription tier""",)
                else:
                    raise api_error

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Process error: {error_msg}")
            return (f"Error: {error_msg}",)

# Node registration
NODE_CLASS_MAPPINGS = {
    "Together Vision 🔍": TogetherVisionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Together Vision 🔍": "Together Vision 🔍"
}

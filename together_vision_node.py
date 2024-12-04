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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TogetherVisionNode:
    def __init__(self):
        self.client = None
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types for the node.
        """
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI image input
                "model_name": (["Paid (Llama-3.2-11B-Vision)", "Free (Llama-Vision-Free)"],),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant that can see and describe images."
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe what you see in this image in detail."
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
                    "max": 100
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_image"
    CATEGORY = "image/text"

    def encode_image(self, image_tensor):
        """
        Converts an image tensor to base64 string.
        """
        try:
            logger.info(f"Image tensor type: {type(image_tensor)}")
            logger.info(f"Image tensor shape: {image_tensor.shape if hasattr(image_tensor, 'shape') else 'No shape'}")

            # Handle different input types
            if isinstance(image_tensor, torch.Tensor):
                logger.info("Converting torch tensor to numpy array")
                image_array = image_tensor.cpu().numpy()
            elif isinstance(image_tensor, np.ndarray):
                logger.info("Input is already a numpy array")
                image_array = image_tensor
            else:
                raise ValueError(f"Unsupported image type: {type(image_tensor)}")

            logger.info(f"Numpy array shape: {image_array.shape}")
            logger.info(f"Numpy array dtype: {image_array.dtype}")

            # Ensure image is in the right format (HWC)
            if len(image_array.shape) == 4:
                logger.info("Removing batch dimension")
                image_array = image_array[0]  # Take first image if batched
            
            if len(image_array.shape) == 3:
                if image_array.shape[0] in [3, 4]:  # If channels first (CHW)
                    logger.info("Converting CHW to HWC format")
                    image_array = np.transpose(image_array, (1, 2, 0))  # Convert to HWC

            logger.info(f"After format conversion - Shape: {image_array.shape}")

            # Handle different channel numbers
            if image_array.shape[-1] == 4:  # RGBA
                logger.info("Converting RGBA to RGB")
                image_array = image_array[..., :3]  # Convert to RGB
            
            # Convert to uint8 if needed
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                logger.info("Converting to uint8")
                image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
            
            logger.info(f"Final array shape: {image_array.shape}, dtype: {image_array.dtype}")
            
            # Create PIL Image
            pil_image = Image.fromarray(image_array)
            logger.info(f"PIL Image size: {pil_image.size}, mode: {pil_image.mode}")
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.info(f"Base64 string length: {len(base64_str)}")
            
            return base64_str
            
        except Exception as e:
            logger.error(f"Error in encode_image: {str(e)}", exc_info=True)
            raise

    def get_api_key(self, provided_key):
        """
        Get API key from either the provided input or .env file
        """
        try:
            if provided_key.strip():
                logger.info("Using provided API key")
                return provided_key.strip()
            
            # Try to load from .env in the same directory as this file
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
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

    def process_image(self, image, model_name, api_key, system_prompt, user_prompt, 
                     temperature, top_p, top_k, repetition_penalty):
        """
        Process the image and generate description using Together API.
        """
        try:
            logger.info("Starting image processing")
            
            # Map friendly model names to actual model IDs
            model_mapping = {
                "Paid (Llama-3.2-11B-Vision)": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                "Free (Llama-Vision-Free)": "meta-llama/Llama-Vision-Free"
            }
            actual_model = model_mapping[model_name]
            
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

            try:
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
                    stream=True
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

            except Exception as api_error:
                error_msg = str(api_error).lower()
                if "rate limit" in error_msg:
                    wait_time = "1 hour"
                    if "try again in" in error_msg:
                        # Try to extract the wait time from the error message
                        time_match = re.search(r"try again in (?:about )?([^:]+)", error_msg)
                        if time_match:
                            wait_time = time_match.group(1)
                    
                    error_message = f"""⚠️ Rate Limit Exceeded

The {model_name} has reached its rate limit.
Please try again in {wait_time}.

Tips:
1. Switch to the paid model for higher limits
2. Wait for the rate limit to reset
3. Use the API key from a different Together AI account"""
                    
                    logger.warning(f"Rate limit exceeded: {error_msg}")
                    return (error_message,)
                else:
                    raise api_error

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

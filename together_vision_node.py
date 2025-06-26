import base64
import io
import os
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from together import Together
import together # For specific exceptions like together.RateLimitError
import httpx # For specific exceptions like httpx.TimeoutException
import torch
import logging
import re
import time
import hashlib # For caching
from typing import Optional, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TogetherVisionNode:
    """
    A custom node for ComfyUI that uses Together AI's Vision models for image description,
    with caching, retries, and more configuration options.
    """
    
    def __init__(self):
        self.client: Optional[Together] = None
        self.last_request_time: float = 0
        self.min_request_interval: float = 1.0  # Minimum seconds between requests (client-side)
        self.cache: dict[str, str] = {}  # Response cache
        self._current_api_key: Optional[str] = None # To track if API key changed
        self._last_seed: Optional[int] = None # To track the last used seed for increment mode
        
    def validate_api_key(self, api_key: str) -> bool:
        """Validate if API key is present and well-formed."""
        if not api_key or len(api_key.strip()) == 0:
            logger.error("API key is missing or empty")
            return False
        # Basic check, can be expanded (e.g., regex for format)
        if not re.match(r"^[a-zA-Z0-9\-_]+$", api_key.strip()):
            logger.warning("API key contains potentially invalid characters. Ensure it's correct.")
        return True
        
    def get_client(self, api_key_input: str) -> Optional[Together]:
        """Get or create Together client with validation and .env fallback."""
        try:
            final_api_key = api_key_input.strip() if api_key_input and api_key_input.strip() else os.getenv('TOGETHER_API_KEY')
            
            if not final_api_key:
                logger.error("No API key provided in node or .env file")
                # Raising here will be caught by process_image and shown to user
                raise ValueError("No API key found. Provide an API key or set TOGETHER_API_KEY in .env.")
            
            if not self.validate_api_key(final_api_key):
                 raise ValueError("Invalid API key format or missing.")

            # Re-initialize client if it's not created or if the API key has changed
            if self.client is None or self._current_api_key != final_api_key:
                os.environ["TOGETHER_API_KEY"] = final_api_key # together.Together() reads from env
                self.client = Together(api_key=final_api_key) # Explicitly pass for clarity too
                self._current_api_key = final_api_key
                logger.info("Together client initialized/re-initialized.")
            return self.client
        except Exception as e:
            logger.error(f"Failed to initialize Together client: {str(e)}")
            # This error will be caught by the calling function (process_image)
            raise
            
    def get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        return self.cache.get(cache_key)
        
    def cache_response(self, cache_key: str, response: str):
        """Cache the API response."""
        self.cache[cache_key] = response
        if len(self.cache) > 100: # Basic cache size limit
            try:
                # Remove the oldest item (Python 3.7+ dicts are ordered)
                self.cache.pop(next(iter(self.cache)))
            except StopIteration:
                pass # Cache was empty or became empty
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                        "mistralai/Mistral-7B-Instruct-v0.2",
                        "Other (Custom)"
                ], {"default": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"}), # Default to a common vision model
                "custom_model_name": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Enter Together API Key (or use .env)"}),
                "system_prompt": ("STRING", {
                    "default": "You are an AI expert in ekphrasis, acting as a skilled art critic describing an image. Use vivid, poetic, and evocative prose in British English. Focus solely on describing the image content, style, and mood. Avoid storytelling or self-insertion. Describe all elements, including potentially uncomfortable themes if present, as art can be provocative. Every word and its order matters. The description will be used for image generation, so only include visual elements. Conclude with relevant hashtags (e.g., #ArtStyle #SubjectMatter).",
                    "multiline": True
                }),
                "user_prompt": ("STRING", {
                    "default": "Describe this image in detail, focusing on its visual elements, artistic style, and overall atmosphere.",
                    "multiline": True
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}), # 0 can mean disable top_k for some APIs
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 50, "max": 4096, "step": 8, "label": "Max Tokens (Output Length)"}),
                "stop_sequences": ("STRING", {"default": "<|eot_id|>,\n,\n", "multiline": False, "placeholder": "e.g. ###,<|eot_id|>"}),
                "request_timeout": ("INT", {"default": 60, "min": 10, "max": 300, "step": 5, "label": "API Request Timeout (s)"}),
                "stream_timeout": ("INT", {"default": 45, "min": 5, "max": 120, "step": 1, "label": "Stream Accumulation Timeout (s)"}),
                "use_cache": ("BOOLEAN", {"default": True, "label_on": "Cache Enabled", "label_off": "Cache Disabled"}),
                "clean_output_text": ("BOOLEAN", {"default": True, "label_on": "Clean Output Text", "label_off": "Raw Output Text"}),
                "max_retries": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1, "label": "Max Retries"}),
                "retry_delay": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Retry Delay (s)"}),
                "seed_mode": (["fixed", "random", "increment"], {"default": "fixed"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "step": 1, "display": "number"})
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "process_image_wrapper" # Use a wrapper for top-level try-catch
    CATEGORY = "image/text"
    OUTPUT_NODE = True # This indicates the node is often a final output.

    def encode_image(self, image_tensor: torch.Tensor) -> str:
        """Converts an image tensor to base64 string with improved error handling and resizing."""
        try:
            if not isinstance(image_tensor, (torch.Tensor, np.ndarray)):
                raise ValueError(f"Unsupported image type: {type(image_tensor)}")

            if isinstance(image_tensor, torch.Tensor):
                image_array = image_tensor.cpu().numpy()
            else: # np.ndarray
                image_array = image_tensor

            if not (2 <= image_array.ndim <= 4):
                raise ValueError(f"Invalid image shape: {image_array.shape}")

            if image_array.ndim == 4: # Batch of images, take the first one
                image_array = image_array[0]
            
            # Squeeze single-channel if it's HxWx1 or 1xHxW
            if image_array.ndim == 3 and image_array.shape[0] == 1: # 1xHxW
                 image_array = np.squeeze(image_array, axis=0)
            elif image_array.ndim == 3 and image_array.shape[2] == 1: # HxWx1
                 image_array = np.squeeze(image_array, axis=2)


            # Normalize if float
            if image_array.dtype in [np.float32, np.float64, torch.float32, torch.float64]:
                if image_array.max() <= 1.0: # Assuming 0-1 range for floats
                    image_array = (image_array * 255)
                image_array = image_array.clip(0, 255).astype(np.uint8)
            elif image_array.dtype != np.uint8:
                raise ValueError(f"Unsupported image dtype: {image_array.dtype}. Must be float or uint8.")

            if image_array.ndim == 2: # Grayscale, convert to RGB
                pil_image = Image.fromarray(image_array, mode='L').convert('RGB')
            elif image_array.ndim == 3:
                if image_array.shape[0] in [3, 4]:  # CHW format
                    image_array = np.transpose(image_array, (1, 2, 0)) # HWC
                
                if image_array.shape[-1] == 4: # RGBA
                    pil_image = Image.fromarray(image_array, mode='RGBA').convert('RGB')
                elif image_array.shape[-1] == 3: # RGB
                    pil_image = Image.fromarray(image_array, mode='RGB')
                else:
                    raise ValueError(f"Unsupported channel count: {image_array.shape[-1]}")
            else:
                raise ValueError(f"Invalid image array dimensions after processing: {image_array.ndim}")

            if pil_image.size[0] * pil_image.size[1] == 0:
                raise ValueError("Invalid image dimensions (0 width or height)")

            max_dimension = 1024 # Max dimension for resizing
            if pil_image.width > max_dimension or pil_image.height > max_dimension:
                pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG") # PNG is lossless and supports transparency if ever needed before RGB convert
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Image encoding error: {str(e)}")
            raise ValueError(f"Failed to encode image: {str(e)}") # Re-raise to be caught by process_image

    def rate_limit_check(self):
        """Implement simple client-side rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_duration = self.min_request_interval - time_since_last
            logger.info(f"Client-side rate limit: sleeping for {sleep_duration:.2f}s")
            time.sleep(sleep_duration)
        self.last_request_time = time.time() # Update after wait, or before call

    def _clean_description(self, text: str) -> str:
        """Applies cleaning regexes to the model output."""
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*(thought|reasoning|reflection|note)[:\s][\s\S]*?(?=\n\n|\Z)', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^\s*Here is the description:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```[\s\S]*?```', '', text, flags=re.MULTILINE) # Remove markdown code blocks
        text = re.sub(r'^\s*\[.*?\]\s*$', '', text, flags=re.MULTILINE) # Remove lines like [system message]
        text = text.lstrip('\n\r ') # Remove leading whitespace more broadly

        # Attempt to remove common conversational preambles/postambles if they are short
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            # Check first paragraph
            first_para_lower = paragraphs[0].lower()
            common_preambles = ["sure, here's a description", "certainly, here is the description", "okay, here you go:", "here is a description"]
            if any(preamble in first_para_lower for preamble in common_preambles) and len(paragraphs[0]) < 150:
                paragraphs.pop(0)
            
            # Check last paragraph for common closing remarks (if any)
            # This is less common for descriptions but could happen
            # For now, focusing on preamble removal.

        return ('\n\n'.join(paragraphs)).strip()

    def process_image_wrapper(self, **kwargs) -> Tuple[str,]:
        """Wrapper to catch all exceptions from process_image and return error string."""
        try:
            return self.process_image(**kwargs)
        except Exception as e:
            logger.error(f"Critical error in TogetherVisionNode: {str(e)}", exc_info=True)
            return (f"Error: Critical node failure: {str(e)}",)

    def process_image(self, model_name: str, custom_model_name: str, api_key: str,
                     system_prompt: str, user_prompt: str,
                     temperature: float, top_p: float, top_k: int, repetition_penalty: float,
                     max_tokens: int, stop_sequences: str,
                     request_timeout: int, stream_timeout: int,
                     use_cache: bool, clean_output_text: bool,
                     max_retries: int, retry_delay: float,
                     seed_mode: str = "fixed", seed: int = 42,
                     image: Optional[torch.Tensor] = None) -> Tuple[str,]:
        """
        Process the image and generate description using Together API.
        """
        if not user_prompt.strip():
            return ("Error: User prompt cannot be empty.",)
        # System prompt can be empty if user intends so, though not recommended for default behavior
        # if not system_prompt.strip():
        #     return ("Error: System prompt cannot be empty.",)

        actual_model = custom_model_name.strip() if model_name == "Other (Custom)" else model_name
        if not actual_model:
            return ("Error: Model name not specified. Select a model or provide a custom model name.",)

        try:
            client = self.get_client(api_key) # Can raise ValueError for API key issues
        except ValueError as ve:
            return (f"Error: API Client Initialization Failed: {str(ve)}",)


        messages = [{"role": "system", "content": system_prompt}]
        base64_image_str = None
        if image is not None:
            try:
                base64_image_str = self.encode_image(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_str}"}}
                    ]
                })
            except ValueError as img_error: # Catch specific error from encode_image
                logger.error(f"Image processing failed: {str(img_error)}")
                return (f"Error processing image: {str(img_error)}",)
        else: # No image provided, text-only query
            if "vision" in actual_model.lower() or "vl" in actual_model.lower() or "llava" in actual_model.lower(): # Heuristic
                 logger.warning(f"Warning: Model '{actual_model}' seems to be a vision model, but no image was provided.")
            messages.append({"role": "user", "content": user_prompt})
        
        # Parse stop sequences
        parsed_stop_sequences = [s.strip() for s in stop_sequences.split(',') if s.strip()] if stop_sequences else None
        
        # Determine seed based on mode
        if seed_mode == "fixed":
            current_seed = seed
        elif seed_mode == "random":
            current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:  # increment
            if self._last_seed is None:
                current_seed = seed
            else:
                current_seed = self._last_seed + 1
        self._last_seed = current_seed

        # Cache Key Generation
        cache_key = None
        if use_cache:
            key_parts = [
                actual_model, system_prompt, user_prompt,
                str(temperature), str(top_p), str(top_k), str(repetition_penalty), str(max_tokens),
                stop_sequences # Add stop_sequences to cache key
            ]
            if base64_image_str:
                key_parts.append(hashlib.sha256(base64_image_str.encode('utf-8')).hexdigest()) # Hash image for shorter key
            
            cache_key_string = ":".join(key_parts)
            cache_key = hashlib.sha256(cache_key_string.encode('utf-8')).hexdigest()
            
            cached_description = self.get_cached_response(cache_key)
            if cached_description is not None:
                logger.info(f"Returning cached response for model {actual_model}.")
                return (cached_description,)

        # API call with retries
        description_text = ""
        last_api_exception = None

        for attempt in range(max_retries + 1):
            try:
                self.rate_limit_check() # Client-side delay

                logger.info(f"Calling Together API (Attempt {attempt + 1}/{max_retries + 1}) - Model: {actual_model}")
                
                api_response_stream = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else None, # Some APIs expect None or omit if 0
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    stop=parsed_stop_sequences,
                    seed=current_seed,
                    stream=True,
                    timeout=float(request_timeout) # httpx timeout
                )

                # Process streamed response
                collected_chunks = []
                stream_start_time = time.time()
                for chunk in api_response_stream:
                    if time.time() - stream_start_time > stream_timeout:
                        logger.error(f"Stream accumulation timed out after {stream_timeout}s.")
                        raise TimeoutError(f"Stream accumulation timed out after {stream_timeout}s.")
                    
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        collected_chunks.append(chunk.choices[0].delta.content)
                
                description_text = "".join(collected_chunks)

                if not description_text.strip():
                    # This could be a valid empty response from the model or an issue.
                    # For a description task, it's usually not desired.
                    logger.warning(f"API returned an empty or whitespace-only response for model {actual_model}.")
                    # If it's truly an issue, it might be caught as an error or fail after retries.
                    # If empty string is valid, this warning is just informational.
                    # Some models might do this if the prompt is unanswerable or content policy.

                break # Success, exit retry loop

            except together.RateLimitError as rle:
                last_api_exception = rle
                logger.error(f"Together API Rate Limit Exceeded: {str(rle)}")
                # Extract wait time if possible
                wait_time_str = "an unspecified period"
                match = re.search(r"(?:try again(?: in| after)|Retry after:) ([\w\s\d.]+?)(?:\.|,|$)", str(rle).lower())
                if match:
                    wait_time_str = match.group(1).strip()
                
                model_type = "free tier" if "free" in actual_model.lower() else "paid tier" # Heuristic
                msg = (f"⚠️ Rate Limit Exceeded for {actual_model} ({model_type}).\n"
                       f"API suggests: Try again in {wait_time_str}.\n"
                       f"Consider waiting, reducing request frequency, or checking your Together AI plan limits.")
                return (msg,) # Exit immediately

            except (together.AuthenticationError, together.PermissionDeniedError) as auth_err:
                last_api_exception = auth_err
                logger.error(f"Together API Authentication/Permission Error: {str(auth_err)}")
                return (f"Error: API Authentication/Permission Failed. Check your API key and account status. Details: {str(auth_err)}",)
            
            except together.BadRequestError as bad_req_err:
                last_api_exception = bad_req_err
                logger.error(f"Together API Bad Request Error: {str(bad_req_err)}")
                error_detail = str(bad_req_err)
                if "Model not found" in error_detail:
                     return (f"Error: Model '{actual_model}' not found or not accessible with your API key. Please check the model name. Details: {error_detail}",)
                return (f"Error: API Bad Request. Check your parameters (e.g., model name, prompts, tokens). Details: {error_detail}",)

            except (httpx.TimeoutException, TimeoutError) as timeout_err: # Catches httpx request timeout & our stream timeout
                last_api_exception = timeout_err
                logger.warning(f"API call attempt {attempt + 1} timed out: {str(timeout_err)}")
                if attempt == max_retries:
                    return (f"Error: API call failed after {max_retries + 1} attempts due to timeout. Last error: {str(timeout_err)}",)
                time.sleep(retry_delay)

            except together.APIConnectionError as conn_err:
                last_api_exception = conn_err
                logger.warning(f"API call attempt {attempt + 1} connection error: {str(conn_err)}")
                if attempt == max_retries:
                     return (f"Error: API call failed after {max_retries + 1} attempts due to connection issue. Last error: {str(conn_err)}",)
                time.sleep(retry_delay)

            except Exception as e: # Catch-all for other API errors or unexpected issues during call
                last_api_exception = e
                logger.error(f"Unexpected error during API call attempt {attempt + 1}: {str(e)}", exc_info=True)
                if attempt == max_retries:
                    return (f"Error: API call failed after {max_retries + 1} attempts. Last error: {str(e)}",)
                # Decide if this generic error is retryable (e.g. 5xx from server)
                # For now, retry generic exceptions.
                time.sleep(retry_delay)
        
        # After retry loop
        if not description_text.strip() and last_api_exception:
            # If all retries failed and we ended up with no text
            logger.error(f"Failed to get a valid response after all retries. Last error: {last_api_exception}")
            return (f"Error: Failed to get response after {max_retries + 1} attempts. Last error: {str(last_api_exception)}",)
        
        if not description_text.strip() and not last_api_exception:
            # If the loop completed, no exception, but text is empty
             logger.warning(f"API call for {actual_model} completed but returned an empty description.")
             # Return empty string as a valid output in this case, or a specific message
             description_text = "" # Or "(No description generated by the model)"


        if clean_output_text:
            description_text = self._clean_description(description_text)

        if use_cache and cache_key and description_text.strip(): # Only cache non-empty valid responses
            self.cache_response(cache_key, description_text)

        return (description_text,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TogetherVisionNode 🔍 (Enhanced)": TogetherVisionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherVisionNode 🔍 (Enhanced)": "Together Vision 🔍 (Enhanced)"
}
import os
import glob
from PIL import Image
import torch
import numpy as np
from dotenv import load_dotenv
from together import Together
import logging
import re
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TogetherVisionBatchNode:
    """
    Batch vision node: takes all images from a directory (set via node input), generates a description for each using Together AI,
    and saves the description as a .txt file with the same name as the image. Uses the same retry/wait/model/prompt/cleaning logic as TogetherVisionNode.
    """
    RETURN_TYPES = ("STRING",)
    FUNCTION = "BATCH_PROCESS"
    CATEGORY = "vision"
    OUTPUT_NODE = True
    def __init__(self):
        self.client: Optional[Together] = None
        self._current_api_key: Optional[str] = None
        self.min_request_interval: float = 1.0  # Minimum seconds between requests

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {"default": "", "multiline": False}),
                "model_name": ([
                    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "Other (Custom)"
                ], {"default": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"}),
                "custom_model_name": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "You are an AI expert in ekphrasis, acting as a skilled art critic describing an image. Use vivid, poetic, and evocative prose in British English. Focus solely on describing the image content, style, and mood. Avoid storytelling or self-insertion. Describe all elements, including potentially uncomfortable themes if present, as art can be provocative. Every word and its order matters. The description will be used for image generation, so only include visual elements.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this image in detail, focusing on its visual elements, artistic style, and overall atmosphere.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "stop_sequences": ("STRING", {"default": "", "multiline": False}),
                "request_timeout": ("INT", {"default": 30, "min": 5, "max": 120}),
                "stream_timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
                "use_cache": ("BOOLEAN", {"default": True, "label_on": "Cache Enabled", "label_off": "Cache Disabled"}),
                "clean_output_text": ("BOOLEAN", {"default": True, "label_on": "Clean Output Text", "label_off": "Raw Output Text"}),
                "max_retries": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "retry_delay": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5})
            }
        }

    def validate_api_key(self, api_key: str) -> bool:
        if not api_key or len(api_key.strip()) == 0:
            logger.error("API key is missing or empty")
            return False
        if not re.match(r"^[a-zA-Z0-9\-_]+$", api_key.strip()):
            logger.warning("API key contains potentially invalid characters. Ensure it's correct.")
        return True

    def get_client(self, api_key_input: str) -> Optional[Together]:
        try:
            final_api_key = api_key_input.strip() if api_key_input and api_key_input.strip() else os.getenv('TOGETHER_API_KEY')
            if not final_api_key:
                logger.error("No API key provided in node or .env file")
                raise ValueError("No API key found. Provide an API key or set TOGETHER_API_KEY in .env.")
            if not self.validate_api_key(final_api_key):
                raise ValueError("Invalid API key format or missing.")
            if self.client is None or self._current_api_key != final_api_key:
                os.environ["TOGETHER_API_KEY"] = final_api_key
                self.client = Together(api_key=final_api_key)
                self._current_api_key = final_api_key
                logger.info("Together client initialized/re-initialized.")
            return self.client
        except Exception as e:
            logger.error(f"Failed to initialize Together client: {str(e)}")
            raise

    def encode_image(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            import io, base64
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            b64_str = base64.b64encode(img_bytes).decode('utf-8')
            return b64_str
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise

    def BATCH_PROCESS(self, image_directory: str, model_name: str, custom_model_name: str, api_key: str,
                      system_prompt: str, user_prompt: str, temperature: float, top_p: float, top_k: int,
                      repetition_penalty: float, max_tokens: int, stop_sequences: str, request_timeout: int,
                      stream_timeout: int, use_cache: bool, clean_output_text: bool, max_retries: int, retry_delay: float):
        """
        Process all images in the given directory, one by one, using the same logic as TogetherVisionNode. Wait for each description before proceeding to the next. Save each as a .txt file with the same name in the same directory.
        """
        client = self.get_client(api_key)
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(image_directory, ext)))
        if not images:
            logger.warning(f"No images found in directory: {image_directory}")
            return
        for image_path in images:
            try:
                logger.info(f"Processing {image_path}")
                # Load image as tensor for compatibility with encode_image logic
                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image)
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # CHW, float32
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dim
                description_tuple = self.process_image(
                    model_name, custom_model_name, api_key,
                    system_prompt, user_prompt, temperature, top_p, top_k, repetition_penalty,
                    max_tokens, stop_sequences, request_timeout, stream_timeout, use_cache,
                    clean_output_text, max_retries, retry_delay, image_tensor
                )
                description = description_tuple[0] if description_tuple else ""
                txt_path = os.path.splitext(image_path)[0] + ".txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(description)
                logger.info(f"Saved description to {txt_path}")
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")

    def process_image(self, model_name: str, custom_model_name: str, api_key: str,
                      system_prompt: str, user_prompt: str, temperature: float, top_p: float, top_k: int,
                      repetition_penalty: float, max_tokens: int, stop_sequences: str, request_timeout: int,
                      stream_timeout: int, use_cache: bool, clean_output_text: bool, max_retries: int, retry_delay: float,
                      image: Optional[torch.Tensor] = None):
        # Direct copy of TogetherVisionNode.process_image, simplified for batch use
        if not user_prompt.strip():
            return ("Error: User prompt cannot be empty.",)
        actual_model = custom_model_name.strip() if model_name == "Other (Custom)" else model_name
        if not actual_model:
            return ("Error: Model name not specified. Select a model or provide a custom model name.",)
        try:
            client = self.get_client(api_key)
        except ValueError as ve:
            return (f"Error: API Client Initialization Failed: {str(ve)}",)
        messages = [{"role": "system", "content": system_prompt}]
        base64_image_str = None
        if image is not None:
            try:
                base64_image_str = self.encode_image_tensor(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_str}"}}
                    ]
                })
            except ValueError as img_error:
                logger.error(f"Image processing failed: {str(img_error)}")
                return (f"Error processing image: {str(img_error)}",)
        else:
            if "vision" in actual_model.lower() or "vl" in actual_model.lower() or "llava" in actual_model.lower():
                logger.warning(f"Warning: Model '{actual_model}' seems to be a vision model, but no image was provided.")
            messages.append({"role": "user", "content": user_prompt})
        parsed_stop_sequences = [s.strip() for s in stop_sequences.split(',') if s.strip()] if stop_sequences else None
        cache_key = None
        if use_cache:
            import hashlib
            key_parts = [
                actual_model, system_prompt, user_prompt,
                str(temperature), str(top_p), str(top_k), str(repetition_penalty), str(max_tokens), stop_sequences
            ]
            if base64_image_str:
                key_parts.append(hashlib.sha256(base64_image_str.encode('utf-8')).hexdigest())
            cache_key_string = ":".join(key_parts)
            cache_key = hashlib.sha256(cache_key_string.encode('utf-8')).hexdigest()
            if hasattr(self, 'cache') and self.cache.get(cache_key) is not None:
                logger.info(f"Returning cached response for model {actual_model}.")
                return (self.cache[cache_key],)
        description_text = ""
        last_api_exception = None
        for attempt in range(max_retries + 1):
            try:
                self.rate_limit_check()
                logger.info(f"Calling Together API (Attempt {attempt + 1}/{max_retries + 1}) - Model: {actual_model}")
                import time
                api_response_stream = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else None,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    stop=parsed_stop_sequences,
                    stream=True,
                    timeout=float(request_timeout)
                )
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
                    logger.warning(f"API returned an empty or whitespace-only response for model {actual_model}.")
                break
            except Exception as e:
                last_api_exception = e
                logger.error(f"API error during attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries:
                    return (f"Error: API call failed after {max_retries + 1} attempts. Last error: {str(e)}",)
                import time
                time.sleep(retry_delay)
        if not description_text.strip() and last_api_exception:
            logger.error(f"Failed to get a valid response after all retries. Last error: {last_api_exception}")
            return (f"Error: Failed to get response after {max_retries + 1} attempts. Last error: {str(last_api_exception)}",)
        if not description_text.strip() and not last_api_exception:
            logger.warning(f"API call for {actual_model} completed but returned an empty description.")
            description_text = ""
        if clean_output_text:
            description_text = self._clean_description(description_text)
        if use_cache and cache_key and description_text.strip() and hasattr(self, 'cache'):
            self.cache[cache_key] = description_text
        return (description_text,)

    def encode_image_tensor(self, image_tensor: torch.Tensor) -> str:
        # Same as TogetherVisionNode.encode_image
        import io, base64
        if not isinstance(image_tensor, (torch.Tensor, np.ndarray)):
            raise ValueError(f"Unsupported image type: {type(image_tensor)}")
        if isinstance(image_tensor, torch.Tensor):
            image_array = image_tensor.cpu().numpy()
        else:
            image_array = image_tensor
        if not (2 <= image_array.ndim <= 4):
            raise ValueError(f"Invalid image shape: {image_array.shape}")
        if image_array.ndim == 4:
            image_array = image_array[0]
        if image_array.ndim == 3 and image_array.shape[0] == 1:
            image_array = np.squeeze(image_array, axis=0)
        elif image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = np.squeeze(image_array, axis=2)
        if image_array.dtype in [np.float32, np.float64, torch.float32, torch.float64]:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255)
            image_array = image_array.clip(0, 255).astype(np.uint8)
        elif image_array.dtype != np.uint8:
            raise ValueError(f"Unsupported image dtype: {image_array.dtype}. Must be float or uint8.")
        if image_array.ndim == 2:
            pil_image = Image.fromarray(image_array, mode='L').convert('RGB')
        elif image_array.ndim == 3:
            if image_array.shape[0] in [3, 4]:
                image_array = np.transpose(image_array, (1, 2, 0))
            if image_array.shape[-1] == 4:
                pil_image = Image.fromarray(image_array, mode='RGBA').convert('RGB')
            elif image_array.shape[-1] == 3:
                pil_image = Image.fromarray(image_array, mode='RGB')
            else:
                raise ValueError(f"Unsupported channel count: {image_array.shape[-1]}")
        else:
            raise ValueError(f"Invalid image array dimensions after processing: {image_array.ndim}")
        if pil_image.size[0] * pil_image.size[1] == 0:
            raise ValueError("Invalid image dimensions (0 width or height)")
        max_dimension = 1024
        if pil_image.width > max_dimension or pil_image.height > max_dimension:
            pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def rate_limit_check(self):
        import time
        current_time = time.time()
        if not hasattr(self, 'last_request_time'):
            self.last_request_time = 0
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_duration = self.min_request_interval - time_since_last
            logger.info(f"Client-side rate limit: sleeping for {sleep_duration:.2f}s")
            time.sleep(sleep_duration)
        self.last_request_time = time.time()

    def _clean_description(self, text: str) -> str:
        # Use the same cleaning logic as TogetherVisionNode
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*(thought|reasoning|reflection|note)[:\s][\s\S]*?(?=\n\n|\Z)', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^\s*Here is the description:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```[\s\S]*?```', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\[.*?\]\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            first_para_lower = paragraphs[0].lower()
            common_preambles = ["sure, here's a description", "certainly, here is the description", "okay, here you go:", "here is a description"]
            if any(preamble in first_para_lower for preamble in common_preambles) and len(paragraphs[0]) < 150:
                paragraphs.pop(0)
        return ('\n\n'.join(paragraphs)).strip()

# Node registration for ComfyUI (if needed)
NODE_CLASS_MAPPINGS = {
    "TogetherVisionBatchNode": TogetherVisionBatchNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherVisionBatchNode": "Together Vision Batch Node"
}

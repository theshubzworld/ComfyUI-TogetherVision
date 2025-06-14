# ComfyUI-TogetherVision

A simple ComfyUI custom node for image description and image generation using Together AI’s Vision models.

## Features

- **Image Description:** Get detailed, AI-generated descriptions of any image.
- **Image Generation:** Use Together AI’s free and paid vision models to generate images.
- **Text-Only Mode:** Works as a regular LLM when no image is provided.
- **Automatic Mode Switching:** The node automatically switches between image and text modes based on your workflow.
- **Easy Setup:** Just add your Together AI API key and start using.

## Supported Models in Vision & Image Generation Nodes

The Together Vision node supports the following vision-capable models:

- deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
- meta-llama/Llama-Vision-Free
- meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo
- Other (Custom, if available)

You can select these models directly from the node's dropdown menu in ComfyUI.

### Non-Vision (Text-Only) Models

The node also supports these non-vision, text-only models:


- meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
- Any other text-only LLMs supported by Together AI

You can use these models for pure text generation tasks when no image is provided.

For image generation, the node uses Together AI's FLUX model.

## Getting Started

1. **Get an API Key:**  
   Sign up at [Together AI](https://together.ai) and create an API key.

2. **Install:**  
   Place this repo in your `ComfyUI/custom_nodes` directory.

3. **Configure:**  
   - Add your API key in a `.env` file as `TOGETHER_API_KEY=your_key_here`
   - Or enter the key directly in the node UI.

4. **Use:**  
   - Add the "Together Vision" node to your ComfyUI workflow.
   - Connect an image for vision tasks, or leave it disconnected for text-only tasks.

## Models

- **Paid:** Llama-3.2-11B-Vision-Instruct-Turbo
- **Free:** Llama-Vision-Free

## Example Prompts

- “Describe this image in detail.”
- “Generate a story about this picture.”
- “What objects are present in this photo?”

## License

MIT License. See [LICENSE](LICENSE) for details.

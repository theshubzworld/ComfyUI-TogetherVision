# ComfyUI-TogetherVision (v2.2.1)

A ComfyUI custom node for image description and image generation using Together AI's Vision models with advanced features including seed control.

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
- lgai/exaone-3-5-32b-instruct
- Any other text-only LLMs supported by Together AI

You can use these models for pure text generation tasks when no image is provided.

For image generation, the node uses Together AI's FLUX model.

## New in v2.2.1

- Added support for lgai/exaone-3-5-32b-instruct model
- Updated model list in the dropdown

## New in v2.2.0

- Added seed mode selection (fixed, random, increment)
- Added seed parameter for reproducible results
- Automatic regeneration on seed/mode changes

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

## Available Models

### Vision Models
- deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
- meta-llama/Llama-Vision-Free
- meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo

### Text-Only Models
- meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
- lgai/exaone-3-5-32b-instruct
- mistralai/Mistral-7B-Instruct-v0.2

You can also use any other model supported by Together AI by selecting "Other (Custom)" and entering the model name.

## Example Prompts

- “Describe this image in detail.”
- “Generate a story about this picture.”
- “What objects are present in this photo?”

## License

MIT License. See [LICENSE](LICENSE) for details.

## ComfyUI Together Vision Node

A custom ComfyUI node that leverages Together AI's Vision models to generate detailed descriptions of images. This node integrates both paid (Llama-3.2-11B-Vision) and free (Llama-Vision-Free) models, allowing users to get high-quality image descriptions directly within their ComfyUI workflows.

### Key Features
- 🎯 Easy-to-use image description node for ComfyUI
- 🔄 Support for both paid and free Together AI Vision models
- 🎚️ Advanced parameter controls (temperature, top_p, top_k, repetition_penalty)
- 🔑 Flexible API key management (via node input or .env file)
- 📝 Customizable system and user prompts
- 🛠️ Comprehensive error handling and logging

### Quick Start
1. Install the node in your ComfyUI custom_nodes directory
2. Add your Together AI API key
3. Connect any image output to the node
4. Get detailed, AI-generated descriptions of your images

Perfect for:
- Content creators needing image descriptions
- Accessibility enhancement
- AI art analysis
- Visual content documentation
- Creative writing inspiration

### Technical Stack
- Python
- Together AI Vision API
- ComfyUI Framework
- PyTorch for tensor handling

### Extended Capabilities

#### New Free Image Generation Node
- 🆓 Introducing a cost-effective Free Image Generation Node
- 🔄 Powered by Flux Schnell Model
- 💡 Enables free image processing and generation
- 🚀 Expand your ComfyUI workflows without additional costs

#### Additional Use Cases
- Cost-effective AI image generation
- Experimental image creation
- Prototype development
- Educational AI exploration

### Model Flexibility
Our node now offers enhanced flexibility with:
- Paid Vision Models: Llama-3.2-11B-Vision-Instruct-Turbo
- Free Vision Models: Llama-Vision-Free
- Image Generation: Flux Schnell Model

**Note**: Free model usage may have limitations compared to paid versions. Performance and output quality can vary.

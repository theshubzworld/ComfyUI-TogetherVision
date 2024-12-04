# ComfyUI-TogetherVision

A custom node for ComfyUI that enables image description using Together AI's Vision models. This node allows you to generate detailed descriptions of images using either the paid or free version of Together AI's Llama Vision models.

![Together Vision Node](placeholder_for_node_screenshot.png)

## Features

- 🖼️ **Image Description**: Generate detailed descriptions of any image using state-of-the-art vision models
- 🤖 **Multiple Models**:
  - Paid Version: Llama-3.2-11B-Vision-Instruct-Turbo
  - Free Version: Llama-Vision-Free
- ⚙️ **Customizable Parameters**:
  - Temperature control
  - Top P sampling
  - Top K sampling
  - Repetition penalty
- 🔑 **Flexible API Key Management**:
  - Direct input in the node
  - Environment variable through .env file
- 📝 **Custom Prompting**:
  - System prompt customization
  - User prompt customization

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mithamunda/ComfyUI-TogetherVision.git
```

2. Install the required dependencies:
```bash
cd ComfyUI-TogetherVision
pip install -r requirements.txt
```

3. Set up your Together AI API key:
   - Option 1: Create a `.env` file in the node directory:
     ```
     TOGETHER_API_KEY=your_api_key_here
     ```
   - Option 2: Input your API key directly in the node

4. Restart ComfyUI

## Usage

1. Add the "Together Vision 🔍" node to your workflow
2. Connect an image output to the node's image input
3. Select your preferred model (Paid or Free)
4. Configure the parameters:
   - Temperature (0.0 - 2.0)
   - Top P (0.0 - 1.0)
   - Top K (1 - 100)
   - Repetition Penalty (0.0 - 2.0)
5. Customize the prompts:
   - System prompt: Sets the behavior of the AI
   - User prompt: Specific instructions for image description

## Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| Temperature | Controls randomness | 0.7 | 0.0 - 2.0 |
| Top P | Nucleus sampling | 0.7 | 0.0 - 1.0 |
| Top K | Top K sampling | 50 | 1 - 100 |
| Repetition Penalty | Prevents repetition | 1.0 | 0.0 - 2.0 |

## Rate Limits

- Free Model (Llama-Vision-Free):
  - Limited requests per hour
  - Consider using the paid version for higher limits
- Paid Model:
  - Higher rate limits
  - Better performance

## Error Handling

The node includes comprehensive error handling and logging:
- API key validation
- Rate limit notifications
- Image processing errors
- API response errors

## Examples

Here are some example prompts you can try:

1. Detailed Description:
```
Describe this image in detail, including colors, objects, and composition.
```

2. Technical Analysis:
```
Analyze this image from a technical perspective, including lighting, composition, and photographic techniques.
```

3. Creative Writing:
```
Write a creative story inspired by this image.
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Together AI for providing the Vision API
- ComfyUI community for the framework and support

## Support

If you encounter any issues or have questions:
1. Check the error logs in ComfyUI
2. Ensure your API key is valid
3. Check Together AI's service status
4. Open an issue on GitHub

---

**Note**: This node requires a Together AI account and API key. You can get one at [Together AI's website](https://together.ai).

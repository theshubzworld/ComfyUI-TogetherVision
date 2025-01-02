from .together_vision_node import TogetherVisionNode
from .together_image_node import TogetherImageNode

NODE_CLASS_MAPPINGS = {
    "TogetherVisionNode": TogetherVisionNode,  # Together Vision node
    "Together Image 🎨": TogetherImageNode,  # Together Image Generation node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Together Image 🎨": "Together Image Generator"
}

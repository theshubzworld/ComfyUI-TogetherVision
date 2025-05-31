from .together_vision_node import TogetherVisionNode
from .together_image_node import TogetherImageNode
from .together_vision_batch_node import TogetherVisionBatchNode

NODE_CLASS_MAPPINGS = {
    "TogetherVisionNode": TogetherVisionNode,  # Together Vision node
    "Together Image 🎨": TogetherImageNode,  # Together Image Generation node
    "TogetherVisionBatchNode": TogetherVisionBatchNode  # Batch Vision node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Together Image 🎨": "Together Image Generator",
    "TogetherVisionBatchNode": "Together Vision Batch Node"
}

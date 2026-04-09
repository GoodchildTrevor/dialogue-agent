from app.tools.base import ExternalToolAdapter, ToolSpec


class ImageGeneratorTool(ExternalToolAdapter):
    spec = ToolSpec(
        name="image_generator",
        description="Generate images through an external image generation API.",
        args_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "size": {"type": "string"},
            },
            "required": ["prompt"],
        },
        layer="external_tool_adapter",
    )
    status_message = "Generating image..."
    invoke_path = "/invoke"

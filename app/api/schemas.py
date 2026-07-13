from pydantic import BaseModel, Field


class UploadedFile(BaseModel):
    file_id: str
    filename: str


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    uploaded_files: list[UploadedFile] = Field(default_factory=list)


class ImageData(BaseModel):
    mime_type: str
    data: str  # base64


class SourceData(BaseModel):
    source: str
    url: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    images: list[ImageData] = Field(default_factory=list)
    sources: list[SourceData] = Field(default_factory=list)

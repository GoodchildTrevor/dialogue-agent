from pydantic import BaseModel, Field


class UploadedFile(BaseModel):
    file_id: str
    filename: str


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    uploaded_files: list[UploadedFile] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str

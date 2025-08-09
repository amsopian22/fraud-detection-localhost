from fastapi import APIRouter

router = APIRouter()

# Placeholder for training endpoints
@router.post("/")
async def train_model():
    return {"message": "Training endpoint not implemented yet."}

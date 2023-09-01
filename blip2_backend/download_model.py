from transformers import AutoProcessor, Blip2ForConditionalGeneration

MODEL_NAME = "Salesforce/blip2-opt-6.7b-coco"
MODEL_CACHE_DIR = "huggingface_cache"


def download_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE_DIR
    )


if __name__ == "__main__":
    download_model()

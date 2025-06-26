#%%
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv
import os
from PIL import Image
import logging
load_dotenv()
# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


#%%
def generate_text(image_path: str, offline: bool = False, ollama_model: str = "granite3.2-vision", openai_model: str = "gpt-4.1-nano-2025-04-14") -> str:
    """
    Generate text from an image using OCR and a language model.
    This function processes an input image file, validates its format and path,
    then performs OCR and extracts text using either an offline Ollama model or
    an online OpenAI model. It is tailored for medical document parsing, returning
    the extracted text as a single continuous block without additional formatting.
    Parameters:
        image_path (str): Path to the image file to be processed.
        offline (bool): If True, use the local Ollama model for inference;
            otherwise, use the OpenAI API.
        ollama_model (str): Name of the local Ollama model to use when offline.
        openai_model (str): Name of the OpenAI model to use when not offline.
    Returns:
        str: The extracted text content from the image.
    Raises:
        FileNotFoundError: If the specified image_path does not exist or is not a file.
        ValueError: If the file extension is not among supported types (png, jpg, jpeg, bmp, gif, tiff)
            or if the file cannot be opened and verified as a valid image.
    """

    # Log the image path being processed
    logger.info("Processing image path: %s", image_path)

    if offline:
        llm = Ollama(
            model=ollama_model,
            request_timeout=600,
            # Manually set the context window to limit memory usage
            context_window=8000,
            temperature=0.1,
            max_tokens=1000,
            top_p=0.9,
        )
    else:
        llm = OpenAI(
            model=openai_model,
            temperature=0.1,
            max_tokens=32768,
        )

    # Validate that the path exists and is a file
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file found at {image_path!r}")

    # Optionally check common image extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    if not image_path.lower().endswith(valid_exts):
        raise ValueError(f"Unsupported file extension. Expected one of {valid_exts}")

    # Attempt to open and verify the image
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Provided file is not a valid image: {e}")

    messages = [
        ChatMessage(
            role="system",
            text="You are a medical document parsing agent. You will be given a document and you need to extract the relevant information from it. Just give the text in a single block without any additional formatting or explanations.",
        ),
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="Perform OCR and extract all text from this document."),
                ImageBlock(path=image_path),
            ],
        ),
    ]

    resp = llm.chat(messages)
    if offline:
        return resp
    else:
        return resp.message.blocks[0].text

# %%

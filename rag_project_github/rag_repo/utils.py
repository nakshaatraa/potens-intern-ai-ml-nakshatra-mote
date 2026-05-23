"""
Utility functions for the RAG system.
Includes language detection and translation capabilities.
"""

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Ensure consistent language detection
DetectorFactory.seed = 0

# Supported languages in our system
SUPPORTED_LANGUAGES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi"
}


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns the language code (e.g., 'en', 'hi', 'mr').
    Defaults to 'en' if detection fails or language is unsupported.
    """
    try:
        lang_code = detect(text)
        if lang_code in SUPPORTED_LANGUAGES:
            return lang_code
        # Fallback to English for unsupported languages
        return "en"
    except Exception:
        return "en"


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text from the source language to English for retrieval.
    """
    if source_lang == "en":
        return text
    
    try:
        translator = GoogleTranslator(source=source_lang, target="en")
        return translator.translate(text)
    except Exception as e:
        print(f"Translation to English failed: {e}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate text from English back to the target language for the final answer.
    """
    if target_lang == "en":
        return text
        
    try:
        translator = GoogleTranslator(source="en", target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation from English failed: {e}")
        return text

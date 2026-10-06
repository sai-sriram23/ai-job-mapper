"""
API Key Manager & Key Pool System
Supports feature-specific API keys (APIKEY_SKILL, APIKEY_COURSEGEN, APIKEY_CAREER, APIKEY_MARKET, APIKEY_JOBSEARCH),
key rotation, dynamic UI overrides, and automatic failovers.
"""

import os
import logging
from typing import List, Callable, Any
from groq import Groq
from tavily import TavilyClient

logger = logging.getLogger(__name__)

# Fallback candidate models for Groq
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b"
]



FEATURE_GROQ_MAPPING = {
    "skill": ["APIKEY_SKILL", "GROQ_API_KEY_SKILL"],
    "coursegen": ["APIKEY_COURSEGEN", "GROQ_API_KEY_COURSEGEN"],
    "career": ["APIKEY_CAREER", "GROQ_API_KEY_CAREER"],
    "market": ["APIKEY_MARKET", "GROQ_API_KEY_MARKET"],
}

FEATURE_TAVILY_MAPPING = {
    "market": ["APIKEY_MARKET", "TAVILY_API_KEY_MARKET"],
    "jobsearch": ["APIKEY_JOBSEARCH", "TAVILY_API_KEY_JOBSEARCH"],
    "resourcesearch": ["APIKEY_RESOURCESEARCH", "TAVILY_API_KEY_RESOURCES"],
}


def load_env():
    """Ensure .env file variables are loaded into os.environ."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path) as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        key, val = line.strip().split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"\'')
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")

load_env()


def get_groq_keys(feature: str = None) -> List[str]:
    """
    Collect all available Groq API keys (starting with 'gsk_'), prioritizing feature-specific keys if feature is specified:
    1. Feature-specific keys (e.g. APIKEY_SKILL, APIKEY_COURSEGEN, APIKEY_CAREER)
    2. Streamlit UI custom keys
    3. Any environment variable starting with GROQ_API_KEY or APIKEY_ with a 'gsk_' prefix
    """
    keys = []
    
    # Helper to check if key is valid Groq key format
    def is_valid_groq(k: str) -> bool:
        return bool(k and k.startswith("gsk_") and not k.startswith("your_"))

    # 1. Feature specific key check
    if feature and feature.lower() in FEATURE_GROQ_MAPPING:
        for var_name in FEATURE_GROQ_MAPPING[feature.lower()]:
            val = os.environ.get(var_name, "").strip()
            if is_valid_groq(val) and val not in keys:
                keys.append(val)

    # 2. Check Streamlit session state
    try:
        import streamlit as st
        if "custom_groq_keys" in st.session_state and st.session_state["custom_groq_keys"]:
            raw_ui = st.session_state["custom_groq_keys"]
            for k in raw_ui.replace("\n", ",").split(","):
                k_clean = k.strip()
                if is_valid_groq(k_clean) and k_clean not in keys:
                    keys.append(k_clean)
    except Exception:
        pass

    # 3. Dynamically scan all environment variables for Groq keys (gsk_)
    env_keys_sorted = sorted([k for k in os.environ.keys() if k.startswith("GROQ_API_KEY") or k.startswith("APIKEY_")])
    for env_var in env_keys_sorted:
        val = os.environ.get(env_var, "")
        if val:
            for k in val.replace("\n", ",").split(","):
                k_clean = k.strip()
                if is_valid_groq(k_clean) and k_clean not in keys:
                    keys.append(k_clean)
                    
    return keys


def get_tavily_keys(feature: str = None) -> List[str]:
    """
    Collect all available Tavily API keys (starting with 'tvly-'), prioritizing feature-specific keys if feature is specified:
    1. Feature-specific keys (e.g. APIKEY_MARKET, APIKEY_JOBSEARCH)
    2. Streamlit UI custom keys
    3. Any environment variable starting with TAVILY_API_KEY or APIKEY_ with a 'tvly-' prefix
    """
    keys = []
    
    # Helper to check if key is valid Tavily key format
    def is_valid_tavily(k: str) -> bool:
        return bool(k and k.startswith("tvly-") and not k.startswith("your_"))

    # 1. Feature specific key check
    if feature and feature.lower() in FEATURE_TAVILY_MAPPING:
        for var_name in FEATURE_TAVILY_MAPPING[feature.lower()]:
            val = os.environ.get(var_name, "").strip()
            if is_valid_tavily(val) and val not in keys:
                keys.append(val)

    # 2. Check Streamlit session state
    try:
        import streamlit as st
        if "custom_tavily_keys" in st.session_state and st.session_state["custom_tavily_keys"]:
            raw_ui = st.session_state["custom_tavily_keys"]
            for k in raw_ui.replace("\n", ",").split(","):
                k_clean = k.strip()
                if is_valid_tavily(k_clean) and k_clean not in keys:
                    keys.append(k_clean)
    except Exception:
        pass

    # 3. Dynamically scan all environment variables for Tavily keys (tvly-)
    env_keys_sorted = sorted([k for k in os.environ.keys() if k.startswith("TAVILY_API_KEY") or k.startswith("APIKEY_")])
    for env_var in env_keys_sorted:
        val = os.environ.get(env_var, "")
        if val:
            for k in val.replace("\n", ",").split(","):
                k_clean = k.strip()
                if is_valid_tavily(k_clean) and k_clean not in keys:
                    keys.append(k_clean)
                    
    return keys



def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "****"
    return key[:4] + "..." + key[-4:]


def execute_groq_with_rotation(prompt: str, temperature: float = 0.3, is_json: bool = True, max_tokens: int = 4096, feature: str = None) -> str:
    """
    Executes Groq API call by rotating through feature-specific API keys and models, falling back to backup key pool.
    """
    keys = get_groq_keys(feature=feature)
    if not keys:
        raise RuntimeError("No Groq API keys configured. Please add an APIKEY_SKILL, APIKEY_COURSEGEN, or GROQ_API_KEY in .env.")
        
    errors = []
    
    for key_idx, key in enumerate(keys):
        masked = _mask_key(key)
        try:
            client = Groq(api_key=key)
        except Exception as client_err:
            logger.warning(f"Failed to instantiate Groq client with key #{key_idx+1} ({masked}): {client_err}")
            errors.append(f"Key #{key_idx+1} ({masked}): {client_err}")
            continue

        for model in GROQ_MODELS:
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if is_json:
                    try:
                        kwargs["response_format"] = {"type": "json_object"}
                        res = client.chat.completions.create(**kwargs)
                        return res.choices[0].message.content
                    except Exception:
                        kwargs.pop("response_format", None)
                        
                res = client.chat.completions.create(**kwargs)
                return res.choices[0].message.content
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Groq Key #{key_idx+1} ({masked}) with model '{model}' failed: {err_msg}")
                errors.append(f"Key #{key_idx+1} ({masked}) [{model}]: {err_msg}")
                if "429" in err_msg or "rate_limit" in err_msg.lower() or "quota" in err_msg.lower() or "401" in err_msg:
                    break
                continue

    raise RuntimeError(f"All Groq API keys ({len(keys)}) failed. Errors:\n" + "\n".join(errors))


def execute_tavily_with_rotation(search_func: Callable[[TavilyClient], Any], feature: str = None) -> Any:
    """
    Executes Tavily API call rotating through feature-specific Tavily keys, falling back to backup pool.
    """
    keys = get_tavily_keys(feature=feature)
    if not keys:
        logger.warning("No Tavily API keys configured. Returning empty search result.")
        return {"results": []}

    errors = []
    for key_idx, key in enumerate(keys):
        masked = _mask_key(key)
        try:
            client = TavilyClient(api_key=key)
            return search_func(client)
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Tavily Key #{key_idx+1} ({masked}) failed: {err_msg}")
            errors.append(f"Key #{key_idx+1} ({masked}): {err_msg}")
            continue

    logger.error(f"All Tavily API keys ({len(keys)}) failed. Errors:\n" + "\n".join(errors))
    return {"results": []}

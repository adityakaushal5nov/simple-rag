import json
import os
import traceback
from abc import ABC

import requests


class BaseLLM(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError

    def chat(self, prompt: str, **kwargs):
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    def __init__(self, *args, **kwargs):
        """Initialize the Gemini language model.
        - api_key: str: The API key for the Gemini API.
        - model_name: str: The Gemini model to use.
        """
        super().__init__(*args, **kwargs)
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or kwargs.get("api_key")
        if not api_key:
            raise ValueError(
                "Please set GOOGLE_API_KEY environment variable or set api_key."
            )
        self.model_name = kwargs.get("model_name", "gemini-2.0-flash")
        self.client = genai.Client(api_key=api_key)
        self._generate_config = types.GenerateContentConfig(temperature=0.0)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model.
        - prompt: str: The prompt to generate text from.
        - retried: int: The number of times the request has been retried.
        """
        from google.genai.errors import ServerError

        retried = kwargs.get("retried", 0)
        if retried < 0:
            raise Exception("Retried too many times.")
        try:
            response = self.client.models.generate_content(
                model=kwargs.get("model_name", self.model_name),
                contents=prompt,
                config=self._generate_config,
            )
            return response.text or ""
        except ServerError:
            print("Retrying 500...", retried)
            return self.generate(prompt, retried=retried)
        except Exception:
            print("An error occurred!")
            traceback.print_exc()
            return self.generate(prompt, retried=retried - 1)

    def chat(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


class OllamaLLM(BaseLLM):
    def __init__(self, *args, **kwargs):
        """Initialize the Ollama language model.
        - base_url: str: The base URL of the Ollama API.
        - model_name: str: The name of the model to use.
        """
        super().__init__(*args, **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.model_name = kwargs.get("model_name")
        if not self.model_name:
            raise ValueError("Please provide a model_name.")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model.
        - prompt: str: The prompt to generate text from.
        - model_name: str: The name of the model to use.
        - stream: bool: Whether to stream the response.
        - format: str: The format of the response.
        - timeout: int: The timeout for the request.
        """
        retried = kwargs.get("retried", 0)
        if retried < 0:
            raise Exception("Giving up after several retries.")
        try:
            data = {
                "model": kwargs.get("model_name", self.model_name),
                "prompt": prompt,
                "stream": kwargs.get("stream", False),
                "format": kwargs.get("format"),
            }
            response = requests.post(
                url=self.base_url + "/api/generate",
                data=json.dumps(data),
                timeout=kwargs.get("timeout", 60),
            )
            response.raise_for_status()
            if format == "json":
                return json.loads(response.json()["response"])
            return response.json()["response"]
        except json.decoder.JSONDecodeError:
            kwargs["retried"] = retried - 1
            return self.generate(prompt, **kwargs)
        except Exception:
            kwargs["retried"] = retried - 1
            return self.generate(prompt, **kwargs)

    def chat(self, prompt: str, **kwargs) -> str:
        # We don't have a plan to implement chat for Ollama.
        raise NotImplementedError

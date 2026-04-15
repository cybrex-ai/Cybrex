from typing import Iterator
from llama_cpp import Llama
from interfaces import CoreInterface

class Module(CoreInterface):
    def __init__(self):
        self.system_prompt = "You are an AI assistant. Keep responses concise."
        self.model_path = "models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=32768,
            offload_kqv=False,
            n_gpu_layers=10,
            main_gpu=0,
            n_threads=4,
            n_threads_batch=4,
            n_batch=512,
            n_ubatch=512,
            use_mmap=True,
            cache=True,
            verbose=False,
        )

    def generate(self, user_input: str, context: list[dict], memories: str) -> Iterator[str]:
        for chunk in self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": f"{self.system_prompt}\n\nRelevant memories from past conversations:\n{memories}"},
                *context,
                {"role": "user", "content": user_input}
            ],
            stream=True,
            temperature=0.45,
            top_p=0.85,
            repeat_penalty=1.1,
            max_tokens=512,
        ):
            token = chunk["choices"][0]["delta"].get("content", "")
            yield token

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
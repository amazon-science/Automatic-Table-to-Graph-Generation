import anthropic

class AnthropicClient:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def get_completion(self, system_prompt, user_prompt, temperature=1.0, model="claude-3-sonnet-20241022", max_tokens=1024):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature
        )
        return response.content

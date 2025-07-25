import requests
import time

class LLMRecap:
    def __init__(self, hf_api_token, model_url="https://api-inference.huggingface.co/models/facebook/bart-large-cnn"):
        self.model_url = model_url
        self.headers = {"Authorization": f"Bearer {hf_api_token}"}

    def summarize_text(self, text: str, chunk_size=3000, sleep_between=2):
        """Divide il testo in chunk e li riassume con Hugging Face"""
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        mini_summaries = []

        for idx, chunk in enumerate(chunks, 1):
            print(f"📝 Riassumendo chunk {idx}/{len(chunks)}...")
            mini_summaries.append(self._summarize_chunk(chunk))
            time.sleep(sleep_between)

        # Se più chunk → riassunto finale dei mini-riassunti
        if len(mini_summaries) > 1:
            combined = " ".join(mini_summaries)
            print("\n🤖 Riassumo i mini-riassunti in un riassunto finale...")
            return self._summarize_chunk(combined)
        else:
            return mini_summaries[0]

    def _summarize_chunk(self, text: str):
        payload = {"inputs": text[:4000]}
        try:
            response = requests.post(self.model_url, headers=self.headers, json=payload, timeout=60)
            result = response.json()
            if isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
            else:
                return f"❌ Errore Hugging Face: {result}"
        except Exception as e:
            return f"❌ Errore rete Hugging Face: {e}"

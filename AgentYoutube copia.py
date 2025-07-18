# YouTube AI Agent - Ricerca, Transcript, Risposta con LLM

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import openai

search_youtube_viff.z

# Inserisci la tua API Key di YouTube
YOUTUBE_API_KEY = "a"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Inserisci la tua API Key di OpenAI
openai.api_key = "a"s
# 1. Ricerca su YouTube
def search_youtube_videos(query, max_results=3):
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        videos.append({"id": video_id, "title": title})
    return videos

# 2. Recupera Transcript
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['it', 'en'])
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"Transcript non disponibile per questo video: {e}"

# 3. Sintesi con OpenAI GPT
def summarize_with_openai(text):
    prompt = f"Riassumi in italiano il seguente contenuto:\n\n{text[:3000]}"  # Limito per token
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un assistente che riassume contenuti di YouTube."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"]

# 4. Interfaccia principale
if __name__ == "__main__":
    query = input("🔍 Inserisci la frase o domanda da cercare su YouTube:\n> ")

    print(f"\n📺 Cerco video per: {query}...\n")
    videos = search_youtube_videos(query)

    if not videos:
        print("❌ Nessun video trovato.")
    else:
        transcripts = []
        for v in videos:
            print(f"▶️ {v['title']} (ID: {v['id']})")
            transcript = get_transcript(v["id"])
            transcripts.append(transcript)
            print("✅ Transcript acquisito\n")

        # Unisco i transcript e passo a GPT
        full_text = " ".join(transcripts)
        print("\n🤖 Sto generando il riassunto con OpenAI...\n")
        summary = summarize_with_openai(full_text)
        print("\n📜 **Riassunto:**\n")
        print(summary)
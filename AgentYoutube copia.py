from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from config import YOUTUBE_API_KEY, OPENAI_API_KEY  # <-- Importiamo le chiavi

# ✅ Setup API
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

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
    return [
        {"id": item["id"]["videoId"], "title": item["snippet"]["title"]}
        for item in response["items"]
    ]

# ✅ 2. Recupera Transcript
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['it', 'en'])
        return " ".join(entry["text"] for entry in transcript)
    except Exception as e:
        print(f"❌ Transcript non disponibile per {video_id}: {e}")
        return None

# ✅ 3. Sintesi con OpenAI GPT
def summarize_with_openai(text, language="italiano"):
    prompt = f"Riassumi in {language} il seguente contenuto:\n\n{text[:4000]}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente che riassume contenuti di YouTube."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=400
    )
    return response.choices[0].message.content

# ✅ 4. Main flow
if __name__ == "__main__":
    query = input("🔍 Inserisci la frase o domanda da cercare su YouTube:\n> ")

    print(f"\n📺 Cerco video per: {query}...\n")
    videos = search_youtube_videos(query)

    if not videos:
        print("❌ Nessun video trovato.")
    else:
        for idx, v in enumerate(videos, start=1):
            print(f"{idx}. ▶️ {v['title']} (https://youtu.be/{v['id']})")

        print("\n✅ Recupero transcript...\n")
        transcripts = []
        for v in videos:
            t = get_transcript(v["id"])
            if t:
                transcripts.append(t)

        if not transcripts:
            print("❌ Nessun transcript disponibile per i video trovati.")
        else:
            full_text = " ".join(transcripts)
            print("🤖 Sto generando il riassunto con OpenAI...\n")
            summary = summarize_with_openai(full_text)
            print("\n📜 **Riassunto finale:**\n")
            print(summary)

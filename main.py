from youtube.YouTubeAdvancedSearch import  YouTubeAdvancedSearch
from llm.LLMRecap import LLMRecap
from config import YOUTUBE_API_KEY, HF_API_TOKEN  # ✅ useremo il token HF dal config

def main():
    # Inserisci le tue chiavi API
 

    yt = YouTubeAdvancedSearch(YOUTUBE_API_KEY)
    recap = LLMRecap(HF_API_TOKEN)

    # 🔍 Ricerca avanzata
    query = input("🔍 Inserisci query di ricerca YouTube: ")
    results = yt.advanced_search(
    query="python tutorial",
    max_results=5,
    order="relevance",
    upload_time=None,
    duration_filter=None
    )

    if not results:
        print("❌ Nessun video trovato.")
        return

    # Mostra risultati
    for i, v in enumerate(results, 1):
        print(f"{i}. ▶️ {v['title']} ({v['duration']}) | 👀 {v['view_count']:,} | 👍 {v['like_count']:,}")
        print(f"   🔗 {v['url']}\n")

    # Scegli un video
    choice = int(input("👉 Scegli un video da riassumere (numero): ")) - 1
    selected_video = results[choice]

    # Recupera transcript
    transcript = yt.get_transcript(selected_video['id'])
    if not transcript:
        print("❌ Nessun transcript disponibile per questo video.")
        return

    print(f"📏 Transcript lunghezza: {len(transcript.split())} parole")

    # Riassunto con Hugging Face
    final_summary = recap.summarize_text(transcript)

    print("\n📜 **RIASSUNTO FINALE**:\n")
    print(final_summary)

if __name__ == "__main__":
    main()

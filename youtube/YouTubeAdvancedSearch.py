import re
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

class YouTubeAdvancedSearch:
    def __init__(self, api_key):
        self.youtube = build("youtube", "v3", developerKey=api_key)
    
    def advanced_search(self, 
                       query="", 
                       channel_id=None,
                       duration_filter=None,   # "short", "medium", "long"
                       upload_time=None,       # "hour", "today", "week", "month", "year"
                       video_type=None,        # "episode", "movie"
                       video_definition=None,  # "high", "standard"
                       video_dimension=None,   # "2d", "3d"
                       order="relevance",      # "date", "rating", "viewCount", "title"
                       safe_search="moderate", # "none", "strict"
                       max_results=10,
                       published_after=None,
                       published_before=None,
                       min_duration=None,      # in seconds
                       max_duration=None,      # in seconds
                       language="en",
                       region_code="US"):
        """Ricerca avanzata YouTube con filtri dettagliati"""
        
        search_params = {
            'part': 'snippet',
            'type': 'video',
            'maxResults': min(max_results, 50),
            'order': order,
            'safeSearch': safe_search,
            'relevanceLanguage': language,
            'regionCode': region_code
        }
        
        if query:
            search_params['q'] = query
        if channel_id:
            search_params['channelId'] = channel_id
        if duration_filter:
            search_params['videoDuration'] = duration_filter
        if upload_time:
            search_params['publishedAfter'] = self._get_time_filter(upload_time)
        if published_after:
            search_params['publishedAfter'] = published_after
        if published_before:
            search_params['publishedBefore'] = published_before
        if video_type:
            search_params['videoType'] = video_type
        if video_definition:
            search_params['videoDefinition'] = video_definition
        if video_dimension:
            search_params['videoDimension'] = video_dimension
        
        try:
            search_response = self.youtube.search().list(**search_params).execute()
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            videos_details = self._get_video_details(video_ids, min_duration, max_duration)
            return videos_details
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def _get_time_filter(self, time_period):
        now = datetime.utcnow()
        time_deltas = {
            'hour': timedelta(hours=1),
            'today': timedelta(days=1),
            'week': timedelta(weeks=1),
            'month': timedelta(days=30),
            'year': timedelta(days=365)
        }
        if time_period in time_deltas:
            past_time = now - time_deltas[time_period]
            return past_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        return None
    
    def _get_video_details(self, video_ids, min_duration=None, max_duration=None):
        if not video_ids:
            return []
        try:
            videos_response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(video_ids)
            ).execute()
            
            detailed_videos = []
            for video in videos_response['items']:
                duration_seconds = self._parse_duration(video['contentDetails']['duration'])
                
                if min_duration and duration_seconds < min_duration:
                    continue
                if max_duration and duration_seconds > max_duration:
                    continue
                
                video_info = {
                    'id': video['id'],
                    'title': video['snippet']['title'],
                    'channel': video['snippet']['channelTitle'],
                    'channel_id': video['snippet']['channelId'],
                    'description': video['snippet']['description'][:200] + "...",
                    'published_at': video['snippet']['publishedAt'],
                    'duration': self._format_duration(duration_seconds),
                    'duration_seconds': duration_seconds,
                    'view_count': int(video['statistics'].get('viewCount', 0)),
                    'like_count': int(video['statistics'].get('likeCount', 0)),
                    'comment_count': int(video['statistics'].get('commentCount', 0)),
                    'thumbnail': video['snippet']['thumbnails']['medium']['url'],
                    'url': f"https://youtu.be/{video['id']}"
                }
                
                detailed_videos.append(video_info)
            
            return detailed_videos
        except Exception as e:
            print(f"❌ Error getting video details: {e}")
            return []
    
    def _parse_duration(self, duration_str):
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    
    def _format_duration(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def get_transcript(self, video_id):
        """Recupera transcript YouTube"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['it', 'en'])
            return " ".join(entry["text"] for entry in transcript)
        except Exception as e:
            print(f"❌ Transcript non disponibile: {e}")
            return None

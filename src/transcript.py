from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


def get_transcript(video_id):
    """
    Fetches the transcript of a YouTube video given its video ID.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        str: The transcript text of the video.
    """
    ytt = YouTubeTranscriptApi()

    try:
        transcript_list = ytt.fetch(video_id, languages=['hi', 'en'])
        transcript_text = " ".join(t.text for t in transcript_list)
    except TranscriptsDisabled: 
        transcript_text = "Transcripts are disabled for this video."   

    return transcript_text 







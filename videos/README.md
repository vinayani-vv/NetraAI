# Videos Folder

Place your video files here for the CCTV streaming application.

## Supported Formats
- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)

## How to Use

1. Place your video file in this folder
2. Rename it to `sample.mp4` or update `app.py` with the filename:
   ```python
   VIDEO_FILE = os.environ.get('VIDEO_FILE', 'videos/your-video-name.mp4')
   ```

## For Cloud Deployment

Upload your video to:
- **GitHub** (in the videos folder)
- **Cloud storage** (S3, Google Cloud Storage) and use the URL
- **External URL** and set the VIDEO_FILE environment variable

## Example Environment Variables

On Render/Railway, set:
- `VIDEO_FILE` = `https://your-storage.com/video.mp4`

Or place the video file in the repository and commit it.

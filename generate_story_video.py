import argparse
import os
from typing import List
from pathlib import Path

from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, AudioFileClip
import pyttsx3
from diffusers import StableDiffusionPipeline
import torch


def generate_images(prompt: str, num_images: int = 6, output_dir: str = "images") -> List[Path]:
    """Generate images from prompt using Stable Diffusion."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    images = []
    for i in range(num_images):
        image = pipe(prompt).images[0]
        img_path = output_path / f"image_{i}.png"
        image.save(img_path)
        images.append(img_path)
    return images


def create_panning_clip(image_path: Path, duration: float = 4.0, frame_size=(1280, 720)) -> ImageClip:
    """Create a left-to-right panning video clip from a still image."""
    img_clip = ImageClip(str(image_path)).resize(height=frame_size[1])
    w, h = img_clip.size
    # Ensure image width covers the frame
    if w < frame_size[0]:
        img_clip = img_clip.resize(width=frame_size[0])
        w, h = img_clip.size
    def make_frame(t):
        x = (w - frame_size[0]) * (t / duration)
        return img_clip.get_frame(t)[
            :, int(x) : int(x) + frame_size[0], :
        ]
    clip = ImageClip(img_clip.get_frame(0), duration=duration).set_make_frame(make_frame)
    return clip.set_duration(duration)


def text_to_speech(text: str, output_file: str = "narration.mp3") -> str:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    return output_file


def create_subtitle_clip(text: str, video_width: int, duration: float) -> TextClip:
    subtitle = TextClip(text, fontsize=40, color="white", stroke_color="black", stroke_width=2, size=(video_width, None), method="caption")
    subtitle = subtitle.set_duration(duration).set_position(("center", "bottom"))
    return subtitle


def main():
    parser = argparse.ArgumentParser(description="Generate video from short story")
    parser.add_argument("story", help="Path to text file containing the story")
    parser.add_argument("--output", default="final_video.mp4", help="Output video file")
    parser.add_argument("--keep_images", action="store_true", help="Keep generated images")
    args = parser.parse_args()

    with open(args.story, "r", encoding="utf-8") as f:
        story_text = f.read().strip()

    images = generate_images(story_text)
    clips = [create_panning_clip(img) for img in images]
    total_duration = sum(clip.duration for clip in clips)
    video = concatenate_videoclips(clips)

    audio_file = text_to_speech(story_text)
    audio_clip = AudioFileClip(audio_file)

    subtitle = create_subtitle_clip(story_text, video.w, total_duration)
    final = CompositeVideoClip([video, subtitle]).set_audio(audio_clip)
    final.write_videofile(args.output, codec="libx264", audio_codec="aac")

    if not args.keep_images:
        for img in images:
            os.remove(img)
        os.remove(audio_file)


if __name__ == "__main__":
    main()

# VideoGeneration

This repository contains a script to generate short videos from a text story.

## Requirements

The script relies on the following Python packages:

- `diffusers` and its dependencies (`torch`, `transformers`)
- `moviepy`
- `pyttsx3`

You will also need access to a GPU for faster image generation. Install the dependencies with:

```bash
pip install diffusers[torch] moviepy pyttsx3
```

Make sure you have the Stable Diffusion weights available (the script uses `runwayml/stable-diffusion-v1-5`).

## Usage

Provide a text file containing your short story:

```bash
python generate_story_video.py story.txt --output story_video.mp4
```

The script will:

1. Generate six images related to the story using Stable Diffusion.
2. Create a panning effect for each image.
3. Convert the story to speech and add it as audio.
4. Overlay the entire story as a subtitle.
5. Concatenate the clips into a 25--30 second video.

Use `--keep_images` if you want to keep the generated images and audio.

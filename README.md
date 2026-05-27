# Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

## AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

## PROBLEM STATEMENT:
Creating realistic images from textual descriptions often requires complex deep-learning architectures and specialized tools. This project addresses this challenge by implementing an image generation application using the Stable Diffusion model and deploying it through an interactive Gradio framework.

## DESIGN STEPS:

### STEP 1:

Model Preparation Use the pre-trained Stable Diffusion model available from Hugging Face's diffusers library. Ensure all dependencies, including GPU acceleration (if available), are set up for optimal performance.

### STEP 2:

Framework Use Gradio to create an interface with: Input: Textbox for entering the prompt. Output: Display panel for the generated image.

### STEP 3:

Workflow Load the Stable Diffusion model and tokenizer. Accept a text prompt from the user via Gradio's input. Preprocess the text and generate an image using Stable Diffusion. Display the generated image in the output panel.

## PROGRAM:
```py
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

# Model ID
MODEL_ID = "runwayml/stable-diffusion-v1-5"

print("Loading Stable Diffusion model...")

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

# Automatically choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipe.to(device)

print(f"Model loaded successfully on {device.upper()}")

# Image generation function
def generate_image(prompt):

    if prompt.strip() == "":
        return None

    image = pipe(
        prompt=prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]

    return image


# Gradio Interface
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter a prompt...\nExample: A futuristic city at sunset",
        label="Prompt"
    ),
    outputs=gr.Image(
        type="pil",
        label="Generated Image"
    ),
    title="Stable Diffusion Image Generator",
    description="Generate images from text prompts using Stable Diffusion and Gradio."
)

# Launch App
if __name__ == "__main__":
    demo.launch()
```

## OUTPUT:
<img width="1919" height="925" alt="image" src="https://github.com/user-attachments/assets/9f5ca070-96a5-42b0-ab46-a22fdfb2b0c7" />


## RESULT:
Thus the image is generated using gradio successfully.

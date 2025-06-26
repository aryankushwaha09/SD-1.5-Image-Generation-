

# Commented out IPython magic to ensure Python compatibility.
from IPython import get_ipython
from IPython.display import display
# %%
# ====== IMPORTS ======
import os
import torch
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import shutil
import glob
from tqdm.auto import tqdm
import gradio as gr

# ====== INSTALL DEPENDENCIES ======
print("[*] Installing dependencies...")
!pip install -q diffusers transformers huggingface_hub accelerate safetensors peft
!pip install -q facexlib gfpgan scipy basicsr realesrgan timm opencv-python numpy Pillow torchvision
!pip install -q gradio

# SwinIR setup
!git clone -q https://github.com/JingyunLiang/SwinIR /content/SwinIR
# %cd /content/SwinIR
!python setup.py develop
!bash /content/SwinIR/download-weights.sh
# %cd /content

# ====== SETUP CODEFORMER ======
if not os.path.exists("/content/CodeFormer"):
    print("[*] Cloning CodeFormer...")
    !git clone -q https://github.com/sczhou/CodeFormer
#     %cd /content/CodeFormer
    !pip install -q -r requirements.txt
    !python basicsr/setup.py develop
    !wget -q https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -P ./weights
#     %cd /content

# Custom ESRGAN model
model_dir = "/content/CodeFormer/weights"
os.makedirs(model_dir, exist_ok=True)

model_url = "https://huggingface.co/uwg/upscale-models/resolve/main/4x_foolhardy_Remacri.pth"
model_path = os.path.join(model_dir, "4x_foolhardy_Remacri.pth")

if not os.path.exists(model_path):
    print("[*] Downloading 4x_foolhardy_Remacri ESRGAN model...")
    os.system(f"wget -q --show-progress -O {model_path} {model_url}")
else:
    print("[*] Model already exists.")
print(f"[*] Model saved at: {model_path}")

# ====== DOWNLOAD Real-ESRGAN x4plus MODEL ======
realesrgan_path = "/content/CodeFormer/weights/RealESRGAN_x4plus.pth"
if not os.path.exists(realesrgan_path):
    print("[*] Downloading RealESRGAN x4plus model...")
    !wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /content/CodeFormer/weights

# ====== FOLDER STRUCTURE ======
os.makedirs("/content/CodeFormer/inputs", exist_ok=True)
os.makedirs("/content/CodeFormer/outputs/final_results", exist_ok=True)
os.makedirs("/content/experiments/pretrained_models", exist_ok=True)

# ====== DOWNLOAD SD MODEL ======
model_path = "/content/Realistic_Vision_V5.1_fp16-no-ema.safetensors"
if not os.path.exists(model_path):
    print("[*] Downloading Realistic Vision checkpoint...")
    !wget -q -O {model_path} https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors
assert os.path.exists(model_path), "[!] Model download failed."

# ====== INIT PIPELINES ======
pipe_txt2img = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None
).to("cuda")

pipe_img2img = StableDiffusionImg2ImgPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None
).to("cuda")

# ====== FACE DETECTION ======
def detect_faces_opencv(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[!] Failed to load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return len(faces)

# ====== DELETE OLD IMAGES ======
def delete_old_images(folder_path):
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if len(image_files) > 1:
        for image_file in image_files[:-1]:
            os.remove(image_file)
            print(f"Deleted: {image_file}")

# ====== GENERATION AND ENHANCEMENT FUNCTION ======
def generate_and_enhance_image(prompt, codeformer_weight, bg_choice, img_height, img_width):
    negative_prompt = (
        "extra fingers, mutated hands, bad anatomy, blurry, ugly, duplicate, missing limbs, disfigured, low quality, distorted face, text, watermark"
    )

    try:
        # Yield initial status for metadata box
        yield None, "[*] Generating low-resolution image..."

        torch.cuda.empty_cache()
        result = pipe_txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=img_height,
            width=img_width,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="pil"
        )
        lowres_latent = result.images[0]

        # Yield status for metadata box
        yield None, f"[*] Latent upscaling to {img_width*2}x{img_height*2}..."
        torch.cuda.empty_cache()
        result_upscaled = pipe_img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=lowres_latent,
            strength=0.25,
            guidance_scale=8.0,
            num_inference_steps=100,
            height=img_height * 2,
            width=img_width * 2
        )
        final_image = result_upscaled.images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = f"/content/CodeFormer/inputs/{timestamp}.png"
        final_image.save(raw_path)
        # Yield status for metadata box
        yield None, f"[*] Saved: {raw_path}"

        faces = detect_faces_opencv(raw_path)
        # Yield status for metadata box
        yield None, f"[*] Faces detected: {faces}"
        if faces == 0:
            # Yield status for metadata box
            yield raw_path, "[!] No faces found. Skipping enhancement."
            return # Return the unenhanced image path

        delete_old_images("/content/CodeFormer/inputs")

        if bg_choice == "SwinIR":
            bg_upsampler = "swinir"
        elif bg_choice == "RealESRGAN_x4plus":
            bg_upsampler = "realesrgan"
        elif bg_choice == "4x_foolhardy_Remacri":
             bg_upsampler = "realesrgan"
        else:
            bg_upsampler = "realesrgan"

        # Yield status for metadata box
        yield None, f"[*] Enhancing with CodeFormer (weight: {codeformer_weight}) + {bg_choice}..."
        enhance_command = (
            f"python /content/CodeFormer/inference_codeformer.py "
            f"-w {codeformer_weight} "
            f"--input_path /content/CodeFormer/inputs "
            f"--output_path /content/CodeFormer/outputs "
            f"--face_upsample "
            f"--bg_upsampler {bg_upsampler}"
        )

        # Use tqdm with the os.system call for a progress bar
        for _ in tqdm(range(1), desc="CodeFormer Enhancement"):
             exit_code = os.system(enhance_command)

        if exit_code != 0:
            # Yield status for metadata box
            yield raw_path, "[!] CodeFormer enhancement failed."
            return # Return the unenhanced image path

        output_path = f"/content/CodeFormer/outputs/final_results/{timestamp}.png"
        if not os.path.exists(output_path):
            # Yield status for metadata box
            yield raw_path, "[!] Output not found. Showing original image."
            return # Return the unenhanced image path

        if bg_choice == "SwinIR":
            # Yield status for metadata box
            yield None, "[*] Applying SwinIR using official test script..."
            # Use tqdm with the os.system call for a progress bar
            for _ in tqdm(range(1), desc="SwinIR Upscaling"):
                !python /content/SwinIR/main_test_swinir.py \
                    --task real_sr \
                    --scale 4 \
                    --model_path /content/SwinIR/experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
                    --folder_lq /content/CodeFormer/outputs/final_results \
                    --folder_gt /content/SwinIR/results/swinir_real_sr_x4/ \
                    --tile 256 --tile_overlap 32

            swinir_out_path = f"/content/results/swinir_real_sr_x4/{timestamp}SwinIR.png"
            if os.path.exists(swinir_out_path):
                shutil.copy(swinir_out_path, output_path)
                # Yield final result and status
                yield output_path, f"[*] Final SwinIR-enhanced image: {output_path}"
                # Clean up intermediate SwinIR output
                shutil.rmtree("/content/results", ignore_errors=True)
                return
            else:
                # Yield final result and status
                yield output_path, "[!] SwinIR output not found."
                return
        else:
            # Yield final result and status
            yield output_path, f"[*] Displaying result from {bg_choice}"
            return


    except torch.cuda.OutOfMemoryError:
        # Yield error status
        yield None, "[!] CUDA OOM. Clearing cache..."
        torch.cuda.empty_cache()
        return
    except Exception as e:
        # Yield error status
        yield None, f"[!] Error: {str(e)}"
        return


# ====== GRADIO INTERFACE ======
if __name__ == "__main__":
    demo = gr.Interface(
        fn=generate_and_enhance_image,
        inputs=[
            gr.Textbox(label="Enter prompt"),
            gr.Slider(minimum=0, maximum=1, value=0.7, label="CodeFormer Weight"), # Slider for CodeFormer weight
            gr.Dropdown(["SwinIR", "RealESRGAN_x4plus", "4x_foolhardy_Remacri"], label="Choose background upsampler model", value="RealESRGAN_x4plus"), # Dropdown for upsampler model
            gr.Slider(minimum=128, maximum=1024, step=64, value=960, label="Image Height"), # Slider for image height
            gr.Slider(minimum=128, maximum=1024, step=64, value=640, label="Image Width") # Slider for image width
        ],
        outputs=[
            gr.Image(label="Generated Image"),  # Output for the image
            gr.Textbox(label="Status/Metadata") # Output for the metadata box
        ],
        title="Image Generation and Enhancement",
        description="Enter a prompt to generate an image and enhance it with CodeFormer and selected upsampler."
    )
    demo.launch()

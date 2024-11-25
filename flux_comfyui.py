import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal


# 기본 환경 설정
image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .apt_install("git")
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.2.7")
    .run_commands("comfy --skip-prompt install --nvidia")
)

# 커스텀 노드 설치
image = (
    image.run_commands("comfy node install was-node-suite-comfyui")
)

# 모델 다운로드 준비
image = (
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(  # needs to be empty for Volume mount to work
        "rm -rf /root/comfy/ComfyUI/models")
)

# Modal 앱 생성
app = modal.App(name="flux_comfyui_app", image=image)

# 모델 저장을 위한 볼륨 설정
vol = modal.Volume.from_name("comfyui-models", create_if_missing=True)

# 모델 다운로드 함수 정의
@app.function(
    volumes={"/root/models": vol},
)
def hf_download(repo_id: str, filename: str, model_type: str):
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=f"/root/models/{model_type}",
    )


# 모델 병렬 다운로드를 위한 엔트리포인트 설정
@app.local_entrypoint()
def download_models():
    models_to_download = [
        ("black-forest-labs/FLUX.1-dev", "ae.safetensors", "vae"),
        ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", "unet"),
        ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors", "clip"),
        ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "clip"),
    ]
    list(hf_download.starmap(models_to_download))

@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="A100",
)
@modal.web_server(8080, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8080", shell=True)

if __name__ == "__main__":
    with app.run():
        ui()

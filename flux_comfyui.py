import modal  # modal 모듈 임포트 추가
import subprocess

# CUDA 기반 환경 설정 및 모델 파일 다운로드
image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .apt_install("git", "wget")
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.2.3")
    .run_commands(
        "comfy --skip-prompt install --nvidia"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors --relative-path models/unet --set-civitai-api-token=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"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors --relative-path models/vae"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors --relative-path models/clip"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors --relative-path models/clip"
    )
)

# Modal 앱 생성
app = modal.App(name="flux_comfyui_app", image=image)

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

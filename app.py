# -*- coding: utf-8 -*-
"""
tpose-server 本体(FastAPI、薄く保つ)。

1枚のキャラクター画像から **Tポーズ4ビュー**(正面 / 背面 / 左前45度 / 右前45度)を
生成する専用サーバ。diffusers-server(統合サーバ)の apps/tpose/ を、必要な
モデル経路(Qwen-Image-Edit)だけを持つ独立リポジトリとして切り出したもの。

エンドポイント:
  POST /api/tpose/generate                 生成ジョブ投入(multipart)
  GET  /api/tpose/jobs/{id}                ジョブ状態(ポーリング)
  GET  /api/tpose/jobs/{id}/images/{k}.png ビュー画像(表示用)
  GET  /api/tpose/jobs/{id}/download/{k}.png ビュー画像(ダウンロード)
  GET  /api/tpose/jobs/{id}/download.zip   4枚 + 入力のZIP
  POST /api/tpose/jobs/{id}/edit           生成後の追加編集(何度でも)
  POST /api/tpose/jobs/{id}/undo           直前の編集を取り消す(1世代)
  GET  /api/tpose/views                    ビューID・プリセット一覧(UI用)
  GET  /api/status                         モデルのロード状態・VRAM
  GET  /api/progress                       生成の進捗(UIのポーリング用)
  POST /api/unload                         VRAM 解放
  POST /api/remove_bg                      任意画像の背景除去(GPU不使用)

GPU 処理は core.gpu.generation_lock 1本で排他される(生成ジョブは blocking 取得)。
static/ を "/" で配信する。既定ポートは 8610(run.sh 参照)。
"""
import io
import os
import time
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

import engine  # noqa: F401  (import 時に core.config 経由で CUDA アロケータ設定を適用)
from core import bg as bg_mod
from core import progress as progress_mod
from tpose import router as tpose_router

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="Tpose 4-view Server")


def _open_upload_image(data: bytes, mode: "str | None" = "RGB") -> Image.Image:
    """アップロード画像を開く共通ヘルパー(EXIF回転の正規化込み)。

    スマホ撮影画像等は「ピクセルは横向きのまま + EXIF Orientation で回転を表現」して
    保存されている。ブラウザのプレビューはEXIFを解釈して正立表示するが、PILの
    `Image.open()` は生ピクセルをそのまま返すため、これを怠ると生成結果だけが90度
    回転する。`ImageOps.exif_transpose()` で回転を実ピクセルに適用する。
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert(mode) if mode else img


# ============================================================================
# ステータス / 進捗 / VRAM 解放
# ============================================================================
@app.get("/api/status")
def status():
    return engine.get_status()


@app.get("/api/progress")
def get_progress():
    """生成中の進捗状態(グローバル1本、core/progress.py)を返す。

    非GPU・排他不要の読み取り専用エンドポイント(core.gpu.generation_lock を一切
    使わない)。生成リクエストと同時に、UIから500ms間隔等でポーリングしてよい。
    """
    return progress_mod.snapshot()


class UnloadRequest(BaseModel):
    target: str = "all"  # "edit" | "all"


@app.post("/api/unload")
def unload(req: UnloadRequest):
    if req.target not in ("edit", "all"):
        raise HTTPException(status_code=400, detail="target は edit / all のいずれかです")
    return engine.unload(req.target)


# ============================================================================
# 背景除去(rembg / animeはCPU、BiRefNet HR Mattingは既定GPU)
# ============================================================================
@app.post("/api/remove_bg")
async def remove_bg(image: UploadFile = File(...), method: str = Form(bg_mod.DEFAULT_BG_METHOD)):
    """任意画像の背景を除去して RGBA PNG を返す(Tポーズ生成とは独立の補助API)。

    method: "rembg"(汎用)| "anime"(アニメ向け)|
      "birefnet_hr_matting"(髪・毛先向け高精度マッティング)
    """
    try:
        src = _open_upload_image(await image.read())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
    start = time.time()
    try:
        out = bg_mod.remove_background(src, method=method)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"背景除去に失敗しました: {exc}")
    name = f"remove_bg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    out.save(os.path.join(OUTPUTS_DIR, name))
    return {
        "mode": "remove_bg",
        "method": bg_mod.resolve_method(method),
        "image_url": f"/outputs/{name}",
        "elapsed_s": time.time() - start,
    }


# ============================================================================
# tpose 本体(ジョブ式API)
# ============================================================================
app.include_router(tpose_router, prefix="/api/tpose")


# ============================================================================
# 生成画像・静的ファイル配信
# ============================================================================
class NoCacheStaticFiles(StaticFiles):
    """静的ファイルにブラウザキャッシュを効かせない(開発中の反映漏れ防止)。"""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )

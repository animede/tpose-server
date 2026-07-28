# -*- coding: utf-8 -*-
"""
tpose アプリ層。

1枚のキャラクター画像から **Tポーズ(両腕を水平に広げた姿勢)の4ビュー**
(正面 / 背面 / 左前45度 / 右前45度)を生成する。image-3d の
マルチビュー入力(Hunyuan3D-2mv)と rig-service(Tポーズ前提の自動リグ・VRM化)
向けの前処理を担う。

移植元は diffusers-server(統合サーバ)の apps/tpose/。設計判断の根拠は
tpose/prompts.py の冒頭コメントに実測値つきで残してある:
  - Multiple-angles LoRA を使わず、engine の通常 Edit を使う
    (Tポーズでは同一性・速度とも優位)。
  - 真横(90度)ビューを持たない(Tポーズの真横投影は構造的に破綻する)。
  - front → 他ビューの2段生成(前面出力を参照画像に連鎖)。

app.py 側は `from tpose import router` して
`app.include_router(router, prefix="/api/tpose")` するだけでよい。
"""
from tpose.jobs import router

__all__ = ["router"]

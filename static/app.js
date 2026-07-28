// Tポーズ4ビュー UI フロントエンド(素のJS、外部ライブラリなし)。
//
// 移植元: diffusers-server の統合評価UI(static/app.js)。このリポジトリは Tポーズ
// 専用の単一ページなので、タブ切替・他タブ(T2I/I2I/Edit/動画...)・ギャラリー・
// LLMプロンプト支援は移植していない。残したのは
//   (1) 汎用ドロップゾーン(input[type=file] を一括でD&D対応にする)
//   (2) ステータスバー(/api/status のポーリングと /api/unload)
//   (3) Tポーズ生成タブのロジック(そのまま)
// の3つ。

// --- 汎用ドロップゾーン化(全タブの input[type="file"] に一括適用) ---
// 目的: 個々のフォームを個別実装せず、DOM上の input[type="file"] を一括で
// ドラッグ&ドロップ対応にする。将来フォームが増えても initGenericDropzones() を
// 呼び直すだけで自動的に対応させたい設計のため、フォーム固有のロジックには
// 一切触れない(既存の FormData 構築・submit/change ハンドラは無変更のまま動く)。
//
// 実装方針:
// - 各 input を「ラッパー(.gd-dropzone-wrap)」で包み、input 自体は
//   position:absolute + opacity:0 で透明化するが、サイズはドロップゾーンいっぱいに
//   広げて残す(input 自身が dragover/drop/click を受け取れるため、ブラウザ標準の
//   ファイル選択ダイアログ挙動をそのまま利用できて実装が単純になる)。
// - ドロップされたファイルは DataTransfer 経由で input.files に代入し、
//   "change" イベントを dispatch する。これにより既存の change ハンドラ
//   (controlnet/inpaint/layered のプレビュー処理等)がそのまま動作する。
// - accept 属性に合わないファイルは拒否し、ドロップゾーン内に一時的にエラー表示する。
// - 既にドロップゾーン化されている input(data-gd-skip="1", 例: charsheetタブは
//   専用の #cs-dropzone 実装を持つため対象外)はスキップする。
const GD_DROPZONE_INIT_FLAG = "gdDropzoneInit";

function _gdFormatBytes(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function _gdAcceptMatches(file, acceptAttr) {
  if (!acceptAttr) return true;
  const patterns = acceptAttr.split(",").map((s) => s.trim()).filter(Boolean);
  if (patterns.length === 0) return true;
  return patterns.some((pattern) => {
    if (pattern.startsWith(".")) {
      return file.name.toLowerCase().endsWith(pattern.toLowerCase());
    }
    if (pattern.endsWith("/*")) {
      const prefix = pattern.slice(0, -1); // "image/"
      return file.type.startsWith(prefix);
    }
    return file.type === pattern;
  });
}

function _gdRenderFileList(listEl, files) {
  listEl.innerHTML = "";
  Array.from(files).forEach((file) => {
    const item = document.createElement("div");
    item.className = "gd-file-item";

    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.className = "gd-file-thumb";
      const reader = new FileReader();
      reader.onload = (e) => {
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
      item.appendChild(img);
    } else {
      const icon = document.createElement("div");
      icon.className = "gd-file-icon";
      icon.textContent = file.type.startsWith("audio/") ? "♪" : file.type.startsWith("video/") ? "▶" : "📄";
      item.appendChild(icon);
    }

    const name = document.createElement("div");
    name.className = "gd-file-name";
    name.textContent = file.name;
    item.appendChild(name);

    const size = document.createElement("div");
    size.className = "gd-file-size";
    size.textContent = _gdFormatBytes(file.size);
    item.appendChild(size);

    listEl.appendChild(item);
  });
}

function _gdShowError(errorEl, message) {
  if (!message) {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
    return;
  }
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
  clearTimeout(errorEl._gdTimer);
  errorEl._gdTimer = setTimeout(() => {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
  }, 4000);
}

function _gdApplyFilesToInput(input, files, labelEl, listEl, errorEl) {
  // accept 属性でフィルタし、不一致があればエラー表示して拒否する。
  const accepted = [];
  const rejected = [];
  Array.from(files).forEach((f) => {
    if (_gdAcceptMatches(f, input.accept)) accepted.push(f);
    else rejected.push(f);
  });

  if (rejected.length > 0) {
    _gdShowError(errorEl, `対応していないファイル形式です: ${rejected.map((f) => f.name).join(", ")}`);
  }
  if (accepted.length === 0) return;

  const dt = new DataTransfer();
  if (input.multiple) {
    accepted.forEach((f) => dt.items.add(f));
  } else {
    dt.items.add(accepted[0]);
  }
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));

  labelEl.classList.add("hidden");
  _gdRenderFileList(listEl, input.files);
}

function _gdSetupOne(input) {
  if (input.dataset[GD_DROPZONE_INIT_FLAG] === "1" || input.dataset.gdSkip === "1") return;
  input.dataset[GD_DROPZONE_INIT_FLAG] = "1";

  const wrap = document.createElement("div");
  wrap.className = "gd-dropzone-wrap";
  wrap.style.position = "relative";

  input.parentNode.insertBefore(wrap, input);

  const zone = document.createElement("div");
  zone.className = "gd-dropzone";

  const label = document.createElement("p");
  label.className = "gd-dropzone-label";
  label.innerHTML = "<strong>ここにファイルをドラッグ&ドロップ</strong><br>またはクリックで選択";

  const fileList = document.createElement("div");
  fileList.className = "gd-file-list";

  const errorEl = document.createElement("p");
  errorEl.className = "gd-dropzone-error hidden";

  zone.appendChild(label);
  zone.appendChild(fileList);
  wrap.appendChild(zone);
  wrap.appendChild(errorEl);
  zone.appendChild(input);

  // 元の input はドロップゾーンいっぱいに重ねて透明化する(クリック・ドラッグ&ドロップの
  // ブラウザ標準挙動をそのまま利用するため input 自体をイベントターゲットにする)。
  input.style.position = "absolute";
  input.style.inset = "0";
  input.style.width = "100%";
  input.style.height = "100%";
  input.style.opacity = "0";
  input.style.cursor = "pointer";
  input.style.margin = "0";
  zone.style.position = "relative";

  function refreshFromInput() {
    if (input.files && input.files.length > 0) {
      label.classList.add("hidden");
      _gdRenderFileList(fileList, input.files);
    } else {
      label.classList.remove("hidden");
      fileList.innerHTML = "";
    }
  }

  input.addEventListener("change", () => {
    // 手動選択(ダイアログ経由)の場合はそのままプレビュー更新のみ行う
    // (change の dispatch はしない = 既存ハンドラの二重発火を避ける)。
    refreshFromInput();
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("gd-dragover");
  });
  zone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    zone.classList.remove("gd-dragover");
  });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("gd-dragover");
    const dtFiles = e.dataTransfer && e.dataTransfer.files;
    if (!dtFiles || dtFiles.length === 0) return;
    _gdApplyFilesToInput(input, dtFiles, label, fileList, errorEl);
  });

  refreshFromInput();
}

function initGenericDropzones(root) {
  const scope = root || document;
  scope.querySelectorAll('input[type="file"]').forEach(_gdSetupOne);
}
initGenericDropzones();

// --- ステータスバー ---
async function refreshStatus() {
  const el = document.getElementById("status-text");
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const vram = data.vram
      ? `VRAM: ${data.vram.allocated_gb.toFixed(1)}GB使用中 / ピーク${data.vram.max_allocated_gb.toFixed(1)}GB / 空き${data.vram.free_gb.toFixed(1)}GB / 総量${data.vram.total_gb.toFixed(1)}GB`
      : "VRAM: CUDA無効";
    const editQuant = data.edit_loaded
      ? `(quant=${data.edit_quant || "bf16/fp8"}${data.edit_lora_available === false ? " LoRA非対応" : ""}${data.edit_lightning_merged ? " Lightning焼込済" : ""})`
      : "";
    const loaded = `Edit:${data.edit_loaded ? "ロード済" + editQuant : "未ロード"}`;
    const busy = data.gpu_busy ? " [GPU使用中]" : "";
    const offload = data.offload_mode ? `offload=${data.offload_mode}` : "offload=未確定(初回ロード時に決定)";
    el.textContent = `${loaded} / ${offload} / ${vram}${busy}`;
  } catch (e) {
    el.textContent = "ステータス取得に失敗しました";
  }
}
document.getElementById("status-refresh-btn").addEventListener("click", refreshStatus);
refreshStatus();
setInterval(refreshStatus, 8000);

// --- 解放ボタン ---
async function unload(target) {
  try {
    const res = await fetch("/api/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    });
    await res.json();
    refreshStatus();
  } catch (e) {
    alert("解放に失敗しました: " + e);
  }
}
document.getElementById("unload-edit-btn").addEventListener("click", () => unload("edit"));
document.getElementById("unload-all-btn").addEventListener("click", () => unload("all"));

// ============================================================================
// Tポーズ4ビュー(tpose)。
// 1枚のキャラクター画像 → Tポーズの正面/背面/左前45度/右前45度(ジョブ式)。
// 移植元: diffusers-server の統合評価UI(static/app.js の tpose タブ)。
// ============================================================================
(() => {
  const form = document.getElementById("tp-form");
  if (!form) return;

  const fileInput = document.getElementById("tp-file-input");
  const tailRefInput = document.getElementById("tp-tailref-input");
  const seedInput = document.getElementById("tp-seed-input");
  const subjectInput = document.getElementById("tp-subject-input");
  const removeBgInput = document.getElementById("tp-removebg-input");
  const clawsInput = document.getElementById("tp-claws-input");
  const recolorInput = document.getElementById("tp-recolor-input");
  const editBox = document.getElementById("tp-edit-box");
  const editInput = document.getElementById("tp-edit-input");
  const editSeed = document.getElementById("tp-edit-seed");
  const editKeepPose = document.getElementById("tp-edit-keeppose");
  const editUseRef = document.getElementById("tp-edit-useref");
  const editBtn = document.getElementById("tp-edit-btn");
  const editAllBtn = document.getElementById("tp-edit-all-btn");
  const editViewsEl = document.getElementById("tp-edit-view-checkboxes");
  const editSelectAllBtn = document.getElementById("tp-edit-select-all-btn");
  const editSelectNoneBtn = document.getElementById("tp-edit-select-none-btn");
  const undoBtn = document.getElementById("tp-undo-btn");
  const editError = document.getElementById("tp-edit-error");
  const palmsInput = document.getElementById("tp-palms-input");
  const pawPadsInput = document.getElementById("tp-pawpads-input");
  const tailPreset = document.getElementById("tp-tail-preset");
  const tailInput = document.getElementById("tp-tail-input");
  const bodyInput = document.getElementById("tp-body-input");
  const costumeInput = document.getElementById("tp-costume-input");
  const bodyPreset = document.getElementById("tp-body-preset");
  const extraInput = document.getElementById("tp-extra-input");
  const generateBtn = document.getElementById("tp-generate-btn");
  const errorEl = document.getElementById("tp-error-message");
  const statusEl = document.getElementById("tp-status-message");
  const progressWrap = document.getElementById("tp-progress");
  const progressFill = document.getElementById("tp-progress-fill");
  const progressCount = document.getElementById("tp-progress-count");
  const viewsGrid = document.getElementById("tp-views-grid");
  const checkboxesEl = document.getElementById("tp-view-checkboxes");
  const zipRow = document.getElementById("tp-zip-row");
  const zipLink = document.getElementById("tp-zip-link");

  let pollTimer = null;
  let busy = false;
  let totalViewCount = 4;
  let currentJobId = null;   // 生成後の編集(/edit・/undo)の対象ジョブ

  const showError = (msg) => {
    errorEl.textContent = msg;
    errorEl.classList.toggle("hidden", !msg);
  };
  const showStatus = (msg) => {
    statusEl.textContent = msg;
    statusEl.classList.toggle("hidden", !msg);
  };

  // ビュー一覧・しっぽプリセットをサーバから取得(既定: 全ビューチェック)
  fetch("/api/tpose/views")
    .then((r) => r.json())
    .then((data) => {
      checkboxesEl.innerHTML = "";
      totalViewCount = data.views.length;
      for (const v of data.views) {
        const label = document.createElement("label");
        label.style.fontWeight = "normal";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = true;
        cb.value = v.key;
        cb.className = "tp-view-cb";
        label.appendChild(cb);
        const note = v.for_3d ? "(3D入力向け)" : "(参考)";
        label.appendChild(document.createTextNode(` ${v.label_ja} ${note}`));
        checkboxesEl.appendChild(label);
      }
      tailPreset.innerHTML = "";
      for (const p of data.tail_presets) {
        const opt = document.createElement("option");
        opt.value = p.value;
        opt.textContent = p.label_ja;
        tailPreset.appendChild(opt);
      }
      bodyPreset.innerHTML = "";
      for (const p of data.body_presets || []) {
        const opt = document.createElement("option");
        opt.value = p.value;
        opt.textContent = p.label_ja;
        bodyPreset.appendChild(opt);
      }
    })
    .catch(() => {
      checkboxesEl.innerHTML = "<p class='form-error'>ビュー一覧の取得に失敗しました</p>";
    });

  // タイルはビューごとに1回だけ作り、以降は中身を更新する(タイルを毎回作り直し、
  // 画像に `?t=Date.now()` を付けるとポーリングのたびに再取得されて**表示が
  // フラッシュする**。ユーザー報告 2026-07-28)。画像の差し替えはサーバが返す
  // `rev`(画像が書き換わった回数)が変わったときだけ行う。
  const tpTiles = new Map();  // key -> {tile, wrap, img, spinner, label, status, dl, dlNobg, actions}
  const TP_STATUS_TEXT = {
    queued: "待機中", running: "生成中", recoloring: "色調整中",
    editing: "編集中", done: "完了", error: "エラー",
  };

  function tpCreateTile(v) {
    const tile = document.createElement("div");
    // view-tile-square: 正方形出力の全身(左右に広げた腕)を切らずに表示する(style.css参照)
    tile.className = "view-tile view-tile-square";
    const wrap = document.createElement("div");
    wrap.className = "view-tile-image-wrap";
    const img = document.createElement("img");
    img.alt = v.label_ja;
    // style.css の .view-tile-image-wrap img は display:none が既定なので、
    // 画像を入れたときに明示的に block にする(charsheetタブと同じ理由)。
    img.style.display = "none";
    wrap.appendChild(img);
    const spinner = document.createElement("div");
    spinner.className = "spinner";
    spinner.style.display = "none";
    wrap.appendChild(spinner);
    tile.appendChild(wrap);
    const label = document.createElement("div");
    label.className = "view-tile-label";
    const note = v.for_3d ? "" : " <span class='en'>(参考)</span>";
    label.innerHTML = `<span class="ja">${v.label_ja}</span> <span class="en">${v.label_en}</span>${note}`;
    tile.appendChild(label);
    const status = document.createElement("div");
    status.className = "view-tile-status";
    tile.appendChild(status);
    const dl = document.createElement("a");
    dl.className = "view-tile-download";
    dl.setAttribute("download", "");
    dl.textContent = "この画像をダウンロード";
    dl.style.display = "none";
    tile.appendChild(dl);
    const dlNobg = document.createElement("a");
    dlNobg.className = "view-tile-download";
    dlNobg.setAttribute("download", "");
    dlNobg.textContent = "背景透過版をダウンロード";
    dlNobg.style.display = "none";
    tile.appendChild(dlNobg);
    const actions = document.createElement("div");
    actions.className = "view-tile-actions";
    actions.style.display = "none";
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "tiny-btn";
    editBtn.textContent = "このビューを編集";
    editBtn.addEventListener("click", () => applyEdit([v.key]));
    actions.appendChild(editBtn);
    const undoOne = document.createElement("button");
    undoOne.type = "button";
    undoOne.className = "tiny-btn secondary";
    undoOne.textContent = "取り消す";
    undoOne.style.display = "none";
    undoOne.addEventListener("click", () => applyUndo([v.key]));
    actions.appendChild(undoOne);
    tile.appendChild(actions);
    viewsGrid.appendChild(tile);
    const entry = { tile, img, spinner, status, dl, dlNobg, actions, undoOne, rev: -1 };
    tpTiles.set(v.key, entry);
    return entry;
  }

  function renderViews(views) {
    tpSyncEditViewChoices(views);
    for (const v of views) {
      const t = tpTiles.get(v.key) || tpCreateTile(v);
      t.status.textContent = TP_STATUS_TEXT[v.status] || v.status;
      t.spinner.style.display =
        (v.status === "running" || v.status === "recoloring" || v.status === "editing")
          ? "block" : "none";
      // 画像は rev が変わったときだけ差し替える(= フラッシュしない)
      const rev = typeof v.rev === "number" ? v.rev : 0;
      if (v.url && rev !== t.rev) {
        t.img.src = v.url + "?v=" + rev;
        t.img.style.display = "block";
        t.rev = rev;
      }
      if (v.download_url) {
        t.dl.href = v.download_url;
        t.dl.style.display = v.status === "done" ? "block" : "none";
      }
      if (v.nobg_download_url) {
        t.dlNobg.href = v.nobg_download_url;
        t.dlNobg.style.display = "block";
      } else {
        t.dlNobg.style.display = "none";
      }
      t.actions.style.display = v.status === "done" ? "flex" : "none";
      t.undoOne.style.display = v.has_prev ? "inline-block" : "none";
    }
    // 別ジョブに切り替わったら(このジョブに無いビューの)タイルを片付ける
    const keys = new Set(views.map((v) => v.key));
    for (const [key, t] of tpTiles) {
      if (!keys.has(key)) {
        t.tile.remove();
        tpTiles.delete(key);
      }
    }
  }

  function tpResetTiles() {
    for (const [, t] of tpTiles) t.tile.remove();
    tpTiles.clear();
    editViewsEl.innerHTML = "";
    editViewKeys = "";
  }

  // --- 編集の対象ビュー選択(生成したビューのぶんだけチェックボックスを作る) ---
  // ジョブのビュー構成が変わったときだけ作り直す(毎回作り直すとポーリングのたびに
  // チェック状態がリセットされてしまうため)。
  let editViewKeys = "";

  function tpSyncEditViewChoices(views) {
    const signature = views.map((v) => v.key).join(",");
    if (signature === editViewKeys) return;
    editViewKeys = signature;
    editViewsEl.innerHTML = "";
    for (const v of views) {
      const label = document.createElement("label");
      label.style.fontWeight = "normal";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.value = v.key;
      cb.className = "tp-edit-view-cb";
      label.appendChild(cb);
      label.appendChild(document.createTextNode(` ${v.label_ja}`));
      editViewsEl.appendChild(label);
    }
  }

  function tpSelectedEditViews() {
    return Array.from(document.querySelectorAll(".tp-edit-view-cb"))
      .filter((cb) => cb.checked)
      .map((cb) => cb.value);
  }

  // --- 生成後の編集(何度でも適用できる汎用Edit)。/edit と /undo を叩く ---
  const showEditError = (msg) => {
    editError.textContent = msg || "";
    editError.classList.toggle("hidden", !msg);
  };

  async function applyEdit(keys) {
    const instruction = (editInput.value || "").trim();
    if (!instruction) { showEditError("修正指示を入力してください"); return; }
    if (!currentJobId || busy) return;
    showEditError("");
    const fd = new FormData();
    fd.append("prompt", instruction);
    fd.append("seed", editSeed.value || "0");
    fd.append("keep_pose", editKeepPose.checked ? "true" : "false");
    fd.append("use_reference", editUseRef && editUseRef.checked ? "true" : "false");
    if (keys && keys.length) fd.append("views", keys.join(","));
    busy = true;
    tpSetEditButtonsDisabled(true);
    showStatus("編集中...");
    try {
      const resp = await fetch(`/api/tpose/jobs/${currentJobId}/edit`, { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      pollTimer = setInterval(() => pollJob(currentJobId), 1500);
      pollJob(currentJobId);
    } catch (err) {
      busy = false;
      tpSetEditButtonsDisabled(false);
      showStatus("");
      showEditError(err.message);
    }
  }

  function tpSetEditButtonsDisabled(disabled) {
    editBtn.disabled = disabled;
    editAllBtn.disabled = disabled;
  }

  async function applyUndo(keys) {
    if (!currentJobId || busy) return;
    showEditError("");
    const fd = new FormData();
    if (keys && keys.length) fd.append("views", keys.join(","));
    try {
      const resp = await fetch(`/api/tpose/jobs/${currentJobId}/undo`, { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      pollJob(currentJobId);
    } catch (err) {
      showEditError(err.message);
    }
  }

  // 「選択したビューに適用」は対象ビューのチェックボックスに従う。
  // 「全ビューに適用」は選択状態に関わらず全ビュー(サーバ側で views 省略 = 全部)。
  editBtn.addEventListener("click", () => {
    const keys = tpSelectedEditViews();
    if (!keys.length) { showEditError("対象ビューを1つ以上選んでください"); return; }
    applyEdit(keys);
  });
  editAllBtn.addEventListener("click", () => applyEdit([]));
  undoBtn.addEventListener("click", () => {
    const keys = tpSelectedEditViews();
    if (!keys.length) { showEditError("対象ビューを1つ以上選んでください"); return; }
    applyUndo(keys);
  });
  editSelectAllBtn.addEventListener("click", () => {
    document.querySelectorAll(".tp-edit-view-cb").forEach((cb) => { cb.checked = true; });
  });
  editSelectNoneBtn.addEventListener("click", () => {
    document.querySelectorAll(".tp-edit-view-cb").forEach((cb) => { cb.checked = false; });
  });

  async function pollJob(jobId) {
    try {
      const resp = await fetch(`/api/tpose/jobs/${jobId}`);
      if (!resp.ok) return;
      const job = await resp.json();
      currentJobId = job.job_id;
      renderViews(job.views);
      if (job.edit_error) showEditError("編集エラー: " + job.edit_error);
      if (job.zip_url) {
        zipLink.href = job.zip_url;
        zipRow.classList.remove("hidden");
      }
      if (job.status === "editing") {
        showStatus("編集中...");
        return;
      }
      if (job.status === "removing_bg") {
        showStatus("背景を削除中...");
        progressWrap.classList.remove("hidden");
        progressCount.textContent = `${job.progress} / ${job.total}`;
        progressFill.style.width = "100%";
        return;
      }
      if (job.status === "running" || job.status === "queued") {
        showStatus(`生成中... (${job.progress} / ${job.total} ビュー完了)`);
        progressWrap.classList.remove("hidden");
        progressCount.textContent = `${job.progress} / ${job.total}`;
        progressFill.style.width = `${(job.progress / Math.max(1, job.total)) * 100}%`;
        return;
      }
      clearInterval(pollTimer);
      pollTimer = null;
      busy = false;
      generateBtn.disabled = false;
      tpSetEditButtonsDisabled(false);
      editBox.classList.remove("hidden");
      progressWrap.classList.add("hidden");
      if (job.status === "done") {
        showStatus("完了しました。image-3d へ渡すのは正面・背面の2枚です。");
      } else {
        showStatus("");
        showError("生成エラー: " + (job.error || "不明なエラー"));
      }
    } catch (_) {
      /* 一時的な失敗は次のポーリングに任せる */
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (busy) return;
    showError("");
    if (!fileInput.files || fileInput.files.length === 0) {
      showError("入力画像を選択してください");
      return;
    }
    const selected = Array.from(document.querySelectorAll(".tp-view-cb:checked")).map((c) => c.value);
    if (selected.length === 0) {
      showError("ビューを1つ以上選択してください");
      return;
    }
    const fd = new FormData();
    fd.append("image", fileInput.files[0]);
    fd.append("seed", seedInput.value || "0");
    fd.append("subject", subjectInput.value);
    if (removeBgInput.checked) fd.append("remove_bg", "true");
    // 背景除去の方式(core/bg.py)。Tポーズはキャラクターが対象なので既定はアニメ向け
    fd.append("bg_method", (document.getElementById("tp-bgmethod-input") || {}).value || "anime");
    fd.append("palms", palmsInput.value);
    fd.append("paw_pads", pawPadsInput.value || "auto");
    fd.append("claws", clawsInput.value);
    // 色調整の指示があれば生成時に渡す(各ビューの生成直後に2パス目が自動で走る)
    if ((recolorInput.value || "").trim()) fd.append("recolor", recolorInput.value.trim());
    // 自由記述があればプリセットより優先
    fd.append("tail", (tailInput.value || "").trim() || tailPreset.value || "");
    // 自由記述があればプリセットより優先
    const bodyValue = (bodyInput.value || "").trim() || bodyPreset.value || "";
    if (bodyValue) fd.append("body", bodyValue);
    // 衣装の背面の見え方(背面ビューにのみ効く)
    if (costumeInput && (costumeInput.value || "").trim()) {
      fd.append("costume", costumeInput.value.trim());
    }
    if ((extraInput.value || "").trim()) fd.append("extra_prompt", extraInput.value.trim());
    if (tailRefInput && tailRefInput.files && tailRefInput.files.length > 0) {
      fd.append("tail_ref", tailRefInput.files[0]);
    }
    // 全選択時は views を省略(サーバ既定=全部)
    if (selected.length < totalViewCount) fd.append("views", selected.join(","));

    busy = true;
    generateBtn.disabled = true;
    zipRow.classList.add("hidden");
    showStatus("ジョブを投入中...");
    try {
      const resp = await fetch("/api/tpose/generate", { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      tpResetTiles();  // 新しいジョブなので前回のタイル(画像)を片付ける
      pollTimer = setInterval(() => pollJob(data.job_id), 1500);
      pollJob(data.job_id);
    } catch (err) {
      busy = false;
      generateBtn.disabled = false;
      showStatus("");
      showError(err.message);
    }
  });
})();

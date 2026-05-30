/**
 * script.js – Narasi Clickbait Detector
 * Frontend. Terintegrasi dengan backend IndoBERT via FastAPI.
 */

// ============================================================
// DATA & STATE
// ============================================================
let searchHistory = [];

// ============================================================
// DETEKSI CLICKBAIT — memanggil backend FastAPI
// ============================================================
async function detectClickbait(headline) {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: headline }), // backend expects "text"
  });

  if (!response.ok) {
    throw new Error("Failed to get prediction from model API.");
  }

  return await response.json();
}

// ============================================================
// HELPER: hitung jumlah kata
// ============================================================
function countWords(str) {
  return str.trim().split(/\s+/).filter(Boolean).length;
}

// ============================================================
// RENDER RESULT CARD
// ============================================================
function renderResult(
  headline,
  isClickbait,
  confidence,
  explanation = [],
  probability = {},
) {
  const resultSection = document.getElementById("resultSection");
  const resultCard = document.getElementById("resultCard");

  const escapedHeadline = escapeHtml(headline);
  const wordCount = countWords(headline);
  const predLabel = isClickbait ? "Clickbait" : "Non-Clickbait";

  // ── Warning headline pendek ───────────────────────────────
  const shortWarningHtml =
    wordCount < 4
      ? `<div class="short-warning">
        <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        <span>Headline terlalu pendek, sehingga hasil prediksi mungkin kurang stabil.
          Masukkan headline yang lebih lengkap untuk hasil yang lebih akurat.</span>
      </div>`
      : "";

  // ── Skor Model + Catatan ──────────────────────────────────
  const scoreHtml = `
    <div class="score-section">
      <div class="score-label-row">
        <span class="score-label-text">Skor Model</span>
        <span class="score-value">${confidence}%</span>
      </div>
      <div class="progress-bar-track">
        <div class="progress-bar-fill ${isClickbait ? "fill-red" : "fill-green"}"
          style="width:${confidence}%"></div>
      </div>
      <p class="score-note">
        Catatan: skor ini berasal dari softmax dan bukan jaminan kebenaran mutlak.
      </p>
    </div>
  `;

  // ── Detail Teknis (probabilitas — lipat) ─────────────────
  // Backend returns keys "Clickbait" and "Non-Clickbait" as decimals 0–1
  const toPercent = (v) => (v !== undefined ? Math.round(v * 100) : "-");
  const clickbaitProb = toPercent(
    probability["Clickbait"] ?? probability.clickbait,
  );
  const nonClickbaitProb = toPercent(
    probability["Non-Clickbait"] ?? probability.non_clickbait,
  );

  const detailHtml = `
    <details class="detail-teknis">
      <summary>Detail Teknis</summary>
      <div class="detail-content">
        <div class="detail-row">
          <span>Probabilitas Clickbait</span>
          <strong>${clickbaitProb}%</strong>
        </div>
        <div class="detail-row">
          <span>Probabilitas Non-Clickbait</span>
          <strong>${nonClickbaitProb}%</strong>
        </div>
      </div>
    </details>
  `;

  // ── LIME (top 5) ─────────────────────────────────────────
  const top5 = explanation.slice(0, 5);

  const limeHtml =
    top5.length > 0
      ? `<div class="lime-box">
        <p class="lime-title">Penjelasan LIME — Top 5 Kata Paling Berpengaruh</p>
        <p class="lime-subtitle">Kata/frasa berikut paling memengaruhi prediksi model:</p>
        <div class="lime-list">
          ${top5
            .map((item) => {
              const weight = Number(item.weight);
              const absW = Number(item.abs_weight);
              const barWidth = Math.min(absW * 1000, 100);
              const isFor =
                item.impact === "supports_prediction" || weight >= 0;
              const dirClass = isFor ? "lime-for" : "lime-against";
              const dirLabel = isFor
                ? `Mendukung prediksi ${predLabel}`
                : `Berlawanan dengan prediksi ${predLabel}`;

              return `
              <div class="lime-item ${dirClass}">
                <span class="lime-word">${escapeHtml(item.word)}</span>
                <div class="lime-bar-wrap">
                  <div class="lime-weight-track">
                    <div class="lime-weight-fill ${dirClass}-fill" style="width:${barWidth}%"></div>
                  </div>
                  <span class="lime-dir-label">${dirLabel}</span>
                </div>
                <span class="lime-contrib" title="Kontribusi LIME: ${weight.toFixed(4)}">
                  ${Math.abs(weight) < 0.001 ? "<0.001" : Math.abs(weight).toFixed(3)}
                </span>
              </div>`;
            })
            .join("")}
        </div>
      </div>`
      : `<div class="lime-box">
        <p class="lime-title">Penjelasan LIME</p>
        <p class="lime-subtitle">Tidak ada explanation yang tersedia.</p>
      </div>`;

  // ── Verdict utama ─────────────────────────────────────────
  if (isClickbait) {
    resultCard.innerHTML = `
      <div class="result-inner">
        <div class="result-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="58" height="58" viewBox="0 0 100 100">
            <polygon points="50,8 95,90 5,90" fill="#f59e0b"/>
            <polygon points="50,12 91,88 9,88" fill="#fbbf24"/>
            <polygon points="50,8 95,90 5,90" fill="#f59e0b" opacity="0.85"/>
            <text x="50" y="76" text-anchor="middle" font-size="44" font-family="sans-serif"
              font-weight="900" fill="white">!</text>
          </svg>
        </div>
        <div class="result-body">
          <div class="result-label">
            <span class="label-waspada">WASPADA</span>
            <span class="label-clickbait">CLICKBAIT</span>
          </div>
          <p class="result-headline">"${escapedHeadline}"</p>
          ${shortWarningHtml}
          <p class="result-desc">
            Judul mengandung sinyal yang dapat memancing rasa penasaran berlebih,
            seperti informasi yang menggantung atau bahasa yang dibuat menarik perhatian.
          </p>
          ${scoreHtml}
          ${detailHtml}
          ${limeHtml}
        </div>
      </div>`;
  } else {
    resultCard.innerHTML = `
      <div class="result-inner">
        <div class="result-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="58" height="58" viewBox="0 0 58 58">
            <circle cx="29" cy="29" r="27" fill="none" stroke="#007a1f" stroke-width="3.5"/>
            <polyline points="16,30 25,40 43,20" fill="none" stroke="#007a1f" stroke-width="4"
              stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="result-body">
          <div class="result-label">
            <span class="label-aman">AMAN DARI POLA CLICKBAIT</span>
          </div>
          <p class="result-headline">"${escapedHeadline}"</p>
          ${shortWarningHtml}
          <p class="result-desc">
            Judul berita cenderung informatif dan tidak menunjukkan pola clickbait yang menyesatkan.
            Model mendeteksinya sebagai konten <strong>Non-Clickbait</strong>.
          </p>
          ${scoreHtml}
          ${detailHtml}
          ${limeHtml}
        </div>
      </div>`;
  }

  resultSection.style.display = "flex";
}

// ============================================================
// TAMBAH KE RIWAYAT
// ============================================================
function addToHistory(headline, isClickbait) {
  const item = { id: Date.now(), headline, isClickbait };
  searchHistory.unshift(item);
  renderHistory();
}

// ============================================================
// RENDER HISTORY
// ============================================================
function renderHistory() {
  const historySection = document.getElementById("historySection");
  const separator = document.getElementById("separator");
  const historyList = document.getElementById("historyList");

  if (searchHistory.length === 0) {
    historySection.style.display = "none";
    separator.style.display = "none";
    return;
  }

  historySection.style.display = "block";
  separator.style.display = "block";

  historyList.innerHTML = searchHistory
    .map(
      (item) => `
    <div class="history-item" data-id="${item.id}">
      <span class="history-badge ${item.isClickbait ? "badge-clickbait" : "badge-aman"}">
        ${item.isClickbait ? "Clickbait" : "Aman"}
      </span>
      <span class="history-headline" title="${escapeHtml(item.headline)}">${escapeHtml(item.headline)}</span>
      <button class="history-remove" data-id="${item.id}" aria-label="Hapus item">
        <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>`,
    )
    .join("");

  historyList.querySelectorAll(".history-remove").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      removeHistoryItem(parseInt(e.currentTarget.dataset.id));
    });
  });
}

// ============================================================
// HAPUS RIWAYAT
// ============================================================
function removeHistoryItem(id) {
  searchHistory = searchHistory.filter((item) => item.id !== id);
  renderHistory();
}

function clearAllHistory() {
  searchHistory = [];
  renderHistory();
}

// ============================================================
// MAIN SEARCH HANDLER
// ============================================================
async function handleSearch() {
  const input = document.getElementById("headlineInput");
  const headline = input.value.trim();
  if (!headline) return;

  try {
    const result = await detectClickbait(headline);
    console.log("API response:", JSON.stringify(result));

    // Backend returns "clickbait"/"non-clickbait" (lowercase)
    const isClickbait = result.prediction_label?.toLowerCase() === "clickbait";

    // Backend returns confidence as decimal 0–1 (e.g. 0.9876 or 1.0), convert to percent
    const rawConf = result.confidence ?? 0;
    const confidencePct =
      rawConf <= 1 ? Math.round(rawConf * 100) : Math.round(rawConf);

    // Backend returns "input_text", not "headline"
    const displayHeadline = result.input_text ?? headline;

    renderResult(
      displayHeadline,
      isClickbait,
      confidencePct,
      result.lime_explanation ?? [],
      result.probabilities ?? {},
    );

    addToHistory(displayHeadline, isClickbait);

    setTimeout(() => {
      document.getElementById("resultSection").scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }, 100);
  } catch (error) {
    console.error(error);
    alert("Gagal menghubungi model. Pastikan backend FastAPI masih berjalan.");
  }
}

// ============================================================
// ESCAPE HTML (XSS guard)
// ============================================================
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ============================================================
// EVENT LISTENERS
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  const searchBtn = document.getElementById("searchBtn");
  const input = document.getElementById("headlineInput");
  const clearBtn = document.getElementById("clearAllBtn");

  if (searchBtn) searchBtn.addEventListener("click", handleSearch);
  if (input)
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleSearch();
    });
  if (clearBtn) clearBtn.addEventListener("click", clearAllHistory);
});

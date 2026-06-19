// ---- config -----------------------------------------------------------
// Change this if the backend isn't running on localhost:8000
const API_BASE = window.PLACEMUX_API_BASE || "http://127.0.0.1:8000";

document.getElementById("apiBaseLabel").textContent = API_BASE;

// ---- tiny fetch helper --------------------------------------------------
async function api(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

// ---- tab navigation -------------------------------------------------------
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("view-" + tab.dataset.view).classList.add("active");
    if (tab.dataset.view === "post") loadCompaniesIntoSelect();
    if (tab.dataset.view === "candidates") loadJobsIntoSelect();
    if (tab.dataset.view === "metrics") loadMetrics();
  });
});

// ---- API health check -------------------------------------------------
async function checkHealth() {
  const dot = document.getElementById("apiDot");
  const text = document.getElementById("apiText");
  try {
    await api("/health");
    dot.className = "dot ok";
    text.textContent = "API connected";
  } catch (e) {
    dot.className = "dot bad";
    text.textContent = "API unreachable — start the backend (see README)";
  }
}

// ===================== VIEW 1: company signup ============================
async function loadCompanyList() {
  const el = document.getElementById("companyList");
  try {
    const companies = await api("/companies");
    el.innerHTML = companies.length
      ? companies.map(c => `<div class="row"><span>${escapeHtml(c.name)}</span><span>${escapeHtml(c.location)}</span></div>`).join("")
      : `<div class="row"><span>No companies yet — sign up the first one.</span></div>`;
  } catch (e) {
    el.innerHTML = `<div class="row"><span>Couldn't load companies (${escapeHtml(e.message)})</span></div>`;
  }
}

document.getElementById("signupForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const hint = document.getElementById("signupHint");
  hint.className = "hint"; hint.textContent = "Signing up…";
  const payload = {
    name: document.getElementById("companyName").value.trim(),
    email: document.getElementById("companyEmail").value.trim(),
    location: document.getElementById("companyLocation").value.trim() || "Remote",
  };
  try {
    const company = await api("/companies/signup", { method: "POST", body: JSON.stringify(payload) });
    hint.className = "hint ok";
    hint.textContent = `✓ ${company.name} is on the marketplace (company_id ${company.id}). Head to "Post a role" next.`;
    document.getElementById("signupForm").reset();
    document.getElementById("companyLocation").value = "Remote";
    loadCompanyList();
  } catch (err) {
    hint.className = "hint bad";
    hint.textContent = `✗ ${err.message}`;
  }
});

// ===================== VIEW 2: post a job =================================
async function loadCompaniesIntoSelect() {
  const sel = document.getElementById("jobCompany");
  try {
    const companies = await api("/companies");
    sel.innerHTML = companies.map(c => `<option value="${c.id}">${escapeHtml(c.name)}</option>`).join("");
  } catch (e) {
    sel.innerHTML = `<option>Couldn't load companies</option>`;
  }
}

function parseSkillString(str) {
  const out = {};
  str.split(",").map(s => s.trim()).filter(Boolean).forEach(pair => {
    const [skill, score] = pair.split(":").map(p => p.trim());
    if (skill && score !== undefined && !isNaN(parseFloat(score))) {
      out[skill.toLowerCase().replace(/\s+/g, "_")] = parseFloat(score);
    }
  });
  return out;
}

document.getElementById("jobForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const hint = document.getElementById("jobHint");
  const companyId = document.getElementById("jobCompany").value;
  if (!companyId) { hint.className = "hint bad"; hint.textContent = "Sign up a company first."; return; }

  const payload = {
    title: document.getElementById("jobTitle").value.trim(),
    location: document.getElementById("jobLocation").value.trim() || "Remote",
    min_experience: parseFloat(document.getElementById("jobExp").value) || 0,
    required_skills: parseSkillString(document.getElementById("jobSkills").value),
    remote_ok: document.getElementById("jobRemote").checked,
  };
  if (Object.keys(payload.required_skills).length === 0) {
    hint.className = "hint bad";
    hint.textContent = "Add at least one skill, e.g. python:60, sql:50";
    return;
  }
  hint.className = "hint"; hint.textContent = "Posting role…";
  try {
    const job = await api(`/companies/${companyId}/jobs`, { method: "POST", body: JSON.stringify(payload) });
    hint.className = "hint ok";
    hint.textContent = `✓ "${job.title}" posted (job_id ${job.id}). Go to "Candidates" to see ranked matches.`;
  } catch (err) {
    hint.className = "hint bad";
    hint.textContent = `✗ ${err.message}`;
  }
});

// ===================== VIEW 3: candidates =================================
async function loadJobsIntoSelect() {
  const sel = document.getElementById("candidateJob");
  try {
    const jobs = await api("/jobs");
    sel.innerHTML = jobs.map(j => `<option value="${j.id}">#${j.id} — ${escapeHtml(j.title)} (${escapeHtml(j.location)})</option>`).join("");
  } catch (e) {
    sel.innerHTML = `<option>Couldn't load jobs</option>`;
  }
}

document.getElementById("loadCandidates").addEventListener("click", async () => {
  const jobId = document.getElementById("candidateJob").value;
  const topK = document.getElementById("candidateTopK").value;
  const resultsEl = document.getElementById("candidateResults");
  if (!jobId) { resultsEl.innerHTML = `<p class="hint bad">No job selected — post one first.</p>`; return; }
  resultsEl.innerHTML = `<p class="hint">Scoring candidates…</p>`;
  try {
    const matches = await api(`/jobs/${jobId}/candidates?top_k=${topK}`);
    resultsEl.innerHTML = matches.map(renderMatchCard).join("");
  } catch (err) {
    resultsEl.innerHTML = `<p class="hint bad">✗ ${escapeHtml(err.message)}</p>`;
  }
});

function renderMatchCard(m) {
  const modelPct = Math.round(m.model_score * 100);
  const basePct = Math.round(m.baseline_score * 100);
  const matchedTags = m.explanation.matched_skills.map(s => `<span class="tag matched">${escapeHtml(s)}</span>`).join("");
  const missingTags = m.explanation.missing_skills.map(s => `<span class="tag missing">${escapeHtml(s)}</span>`).join("");
  return `
  <div class="match-card">
    <div class="rank-seal">${m.rank}</div>
    <div class="match-body">
      <h3>${escapeHtml(m.student_name)} <span style="font-family:var(--mono); font-size:12px; color:var(--ink-soft); font-weight:400;">· student_id ${m.student_id}</span></h3>
      <p class="plain-en">${escapeHtml(m.explanation.plain_english)} ${escapeHtml(m.explanation.location_fit)}.</p>
      <div class="skill-tags">${matchedTags}${missingTags}</div>
    </div>
    <div class="score-block">
      <div class="score-row">
        <span class="score-label">Model</span>
        <div class="score-bar-track"><div class="score-bar-fill model" style="width:${modelPct}%"></div></div>
        <span class="score-num">${modelPct}%</span>
      </div>
      <div class="score-row">
        <span class="score-label">Baseline</span>
        <div class="score-bar-track"><div class="score-bar-fill baseline" style="width:${basePct}%"></div></div>
        <span class="score-num">${basePct}%</span>
      </div>
    </div>
  </div>`;
}

// ===================== VIEW 4: metrics =====================================
async function loadMetrics() {
  const card = document.getElementById("metricsCard");
  const coefCard = document.getElementById("coefCard");
  card.innerHTML = "Loading metrics…";
  try {
    const m = await api("/metrics");
    card.innerHTML = `
      <h2>Held-out test set — model vs. baseline</h2>
      <table class="metric-table">
        <thead><tr><th></th><th>Baseline (skill overlap)</th><th>Trained model</th></tr></thead>
        <tbody>
          <tr><td>Precision</td><td>${fmtPct(m.baseline.precision)}</td><td class="${m.model.precision >= m.baseline.precision ? 'win' : ''}">${fmtPct(m.model.precision)}</td></tr>
          <tr><td>Recall</td><td>${fmtPct(m.baseline.recall)}</td><td class="${m.model.recall >= m.baseline.recall ? 'win' : ''}">${fmtPct(m.model.recall)}</td></tr>
          <tr><td>False-positive rate</td><td>${fmtPct(m.baseline.false_positive_rate)}</td><td class="${m.model.false_positive_rate <= m.baseline.false_positive_rate ? 'win' : ''}">${fmtPct(m.model.false_positive_rate)}</td></tr>
          <tr><td>ROC AUC</td><td>${m.baseline.roc_auc}</td><td class="${m.model.roc_auc >= m.baseline.roc_auc ? 'win' : ''}">${m.model.roc_auc}</td></tr>
          <tr><td>Accuracy</td><td>${fmtPct(m.baseline.accuracy)}</td><td class="${m.model.accuracy >= m.baseline.accuracy ? 'win' : ''}">${fmtPct(m.model.accuracy)}</td></tr>
        </tbody>
      </table>
      <div class="lift-banner">
        Trained on ${m.trained_on_pairs} student↔job pairs · evaluated on ${m.test_set_size} held-out pairs (never seen during training)
        · model precision lift over baseline: <strong>${m.lift_over_baseline_pct}%</strong> · model version <code>${m.model_version}</code>
      </div>
    `;
    if (m.feature_coefficients) {
      const max = Math.max(...Object.values(m.feature_coefficients).map(v => Math.abs(v)));
      coefCard.innerHTML = `
        <h2>Why the model decides what it decides</h2>
        <p class="lede" style="font-size:13.5px; margin-bottom:18px;">Logistic regression coefficients — each bar is how much that feature pushes the match probability up (teal) or down (rust), holding the rest fixed.</p>
        ${Object.entries(m.feature_coefficients).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).map(([name, val]) => {
          const widthPct = (Math.abs(val) / max) * 50;
          const cls = val >= 0 ? "pos" : "neg";
          return `<div class="coef-bar-row">
            <span>${escapeHtml(name)}</span>
            <div class="coef-track"><div class="coef-mid"></div><div class="coef-fill ${cls}" style="width:${widthPct}%"></div></div>
            <span>${val.toFixed(2)}</span>
          </div>`;
        }).join("")}
      `;
    }
  } catch (err) {
    card.innerHTML = `<p class="hint bad">✗ ${escapeHtml(err.message)}</p>`;
  }
}

function fmtPct(v) { return Math.round(v * 100) + "%"; }
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- boot ---------------------------------------------------------------
checkHealth();
loadCompanyList();

"use strict";

const MAX_FILE_BYTES = 2_000_000;
const ALLOWED_EXTENSIONS = [".txt", ".log", ".json"];
const requestedApiBase = new URLSearchParams(window.location.search).get("api");
const localLiveServer = ["localhost", "127.0.0.1"].includes(window.location.hostname)
  && window.location.port !== "8000";
const API_BASE = (requestedApiBase || (localLiveServer ? "http://127.0.0.1:8000" : ""))
  .replace(/\/+$/, "");
const apiUrl = (path) => `${API_BASE}${path}`;
const $ = (id) => document.getElementById(id);

const input = $("incident-input");
const analyzeBtn = $("analyze-btn");
let assistantContextToken = null;
let analyzedInputText = null;
let assistantHistory = [];

// ---------- DOM helpers (all dynamic text goes through textContent: no HTML injection) ----------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

/** Render text; `backtick` spans become command blocks flagged for human review. */
function renderRichText(target, text) {
  const parts = String(text).split(/`([^`]+)`/);
  let hasCommand = false;
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      target.appendChild(el("code", "cmd", part));
      hasCommand = true;
    } else if (part) {
      target.appendChild(document.createTextNode(part));
    }
  });
  if (hasCommand) target.appendChild(el("span", "cmd-flag", "Suggested command - human review required"));
}

function fillList(listId, items, rich = false, emptyText = "None reported.") {
  const list = $(listId);
  clear(list);
  const values = items && items.length ? items : [emptyText];
  values.forEach((item) => {
    const li = el("li");
    rich ? renderRichText(li, item) : (li.textContent = item);
    list.appendChild(li);
  });
}

function fillChips(containerId, items, emptyText = "None detected") {
  const box = $(containerId);
  clear(box);
  if (!items || !items.length) { box.appendChild(el("span", "muted", emptyText)); return; }
  items.forEach((item) => box.appendChild(el("span", "chip", item)));
}

// ---------- UI state ----------

function showState(name) {
  ["empty", "loading", "error"].forEach((s) => { $(`state-${s}`).hidden = s !== name; });
}

function setBusy(busy) {
  analyzeBtn.disabled = busy;
  analyzeBtn.textContent = busy ? "Analyzing..." : "Analyze Incident";
}

function updateCharCount() {
  $("char-count").textContent = `${input.value.length.toLocaleString()} characters`;
}

function severityClass(label) {
  const key = String(label || "unknown").toLowerCase();
  return ["critical", "high", "moderate", "medium", "low"].includes(key) ? `sev-${key}` : "sev-unknown";
}

// ---------- Rendering ----------

function renderSeverity(severityAssessment) {
  const panel = $("severity-index");
  if (!severityAssessment) {
    panel.hidden = true;
    return;
  }
  const score = severityAssessment.score;
  const sev = severityClass(severityAssessment.level);
  $("severity-score").textContent = score === null ? "—" : score;
  const badge = $("severity-badge");
  badge.textContent = score === null ? "Insufficient evidence" : severityAssessment.level;
  badge.className = `badge ${sev}`;
  const meter = $("severity-meter");
  meter.className = `meter-fill ${sev}`;
  meter.style.width = `${score ?? 0}%`;
  const meterContainer = meter.parentElement;
  if (score === null) meterContainer.removeAttribute("aria-valuenow");
  else meterContainer.setAttribute("aria-valuenow", String(score));
  $("severity-basis").textContent = severityAssessment.basis;
  fillList("severity-evidence", severityAssessment.evidence, false,
    "No explicit severity-driving signals were detected in the submitted text.");
  panel.hidden = false;
}

function renderAnalysis(analysis, signals) {
  $("ai-severity").textContent = analysis.severity || "Unknown";
  $("confidence-score").textContent = analysis.confidence_score;
  $("confidence-meter").style.width = `${analysis.confidence_score}%`;

  $("summary").textContent = analysis.incident_summary || "Not provided.";
  $("root-cause").textContent = analysis.root_cause || "Not provided.";
  $("reasoning").textContent = analysis.reasoning || "Not provided.";
  fillChips("components", analysis.affected_components);
  fillList("symptoms", analysis.observed_symptoms);
  fillList("resolution", analysis.recommended_resolution, true, "No resolution steps were suggested.");
  fillList("prevention", analysis.preventive_actions);
  fillList("verification", analysis.verification_steps, true);
  renderFacts(signals);
  $("analysis").hidden = false;
}

function renderFacts(signals) {
  const facts = $("facts");
  clear(facts);
  const rows = signals ? [
    ["Service", signals.service], ["Environment", signals.environment], ["Category", signals.category],
    ["Error signals", signals.error_types], ["Components", signals.affected_components],
    ["Severity indicators", signals.severity_indicators],
  ] : [];
  rows.forEach(([label, value]) => {
    const text = Array.isArray(value) ? value.join(", ") : value;
    facts.appendChild(el("dt", "", label));
    facts.appendChild(el("dd", "", text || "not detected"));
  });
  if (signals && signals.redactions) {
    facts.appendChild(el("dt", "", "Masked secrets"));
    facts.appendChild(el("dd", "", `${signals.redactions} value(s) redacted before analysis`));
  }
  if (signals && signals.truncated) {
    facts.appendChild(el("dt", "", "Truncated"));
    facts.appendChild(el("dd", "", "Input was long; the middle section was omitted from the analysis request."));
  }
}

function evidenceSection(title, items) {
  const wrap = el("div");
  wrap.appendChild(el("h5", "", title));
  if (Array.isArray(items)) {
    const ul = el("ul");
    (items.length ? items : ["(none recorded)"]).forEach((i) => ul.appendChild(el("li", "", i)));
    wrap.appendChild(ul);
  } else {
    wrap.appendChild(el("div", "", items || "(not recorded)"));
  }
  return wrap;
}

function renderEvidence(incidents, citedIds) {
  const panel = $("evidence-panel");
  const list = $("evidence");
  clear(list);
  panel.hidden = false;
  $("evidence-empty").hidden = incidents.length > 0;
  incidents.forEach((item) => {
    const card = el("div", "evidence-item");
    const top = el("div", "evidence-top");
    top.appendChild(el("span", "evidence-id", item.incident_id));
    top.appendChild(el("span", `badge ${severityClass(item.severity)}`, item.severity));
    if (citedIds.includes(item.incident_id)) top.appendChild(el("span", "cited", "Referenced in assessment"));
    card.appendChild(top);
    card.appendChild(el("div", "evidence-title", item.title));

    const meta = el("div", "evidence-meta");
    const sim = el("span");
    sim.appendChild(document.createTextNode(`Similarity: ${(item.similarity * 100).toFixed(1)}% `));
    const bar = el("span", "sim-bar");
    const fill = el("span");
    fill.style.width = `${Math.round(item.similarity * 100)}%`;
    bar.appendChild(fill);
    sim.appendChild(bar);
    meta.appendChild(sim);
    meta.appendChild(el("span", "", `Category: ${item.category}`));
    meta.appendChild(el("span", "", `Service: ${item.service}`));
    card.appendChild(meta);

    const details = el("details");
    details.appendChild(el("summary", "", "Show historical details"));
    details.appendChild(evidenceSection("Symptoms", item.symptoms));
    details.appendChild(evidenceSection("Root cause", item.root_cause));
    details.appendChild(evidenceSection("Resolution (historical)", item.resolution));
    details.appendChild(evidenceSection("Preventive actions", item.preventive_actions));
    card.appendChild(details);
    list.appendChild(card);
  });
}

function renderWarnings(warnings) {
  const panel = $("warnings-panel");
  const list = $("warnings");
  clear(list);
  const items = warnings && warnings.length ? warnings : ["Human review required."];
  items.forEach((w) => list.appendChild(el("li", "", w)));
  panel.hidden = false;
}

function appendAssistantMessage(role, content) {
  const messages = $("assistant-messages");
  const message = el("div", `assistant-message assistant-${role}`);
  message.appendChild(el("div", "assistant-message-role", role === "user" ? "You" : "Assistant"));
  message.appendChild(el("p", "assistant-message-content", content));
  messages.appendChild(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

function resetAssistant() {
  assistantContextToken = null;
  analyzedInputText = null;
  assistantHistory = [];
  $("assistant-dock").hidden = true;
  $("assistant-panel").hidden = true;
  $("assistant-launcher").setAttribute("aria-expanded", "false");
  $("assistant-question").disabled = false;
  $("assistant-send").disabled = false;
  clear($("assistant-messages"));
}

function invalidateAssistantContext() {
  if (!assistantContextToken || input.value.trim() === analyzedInputText) return;
  assistantContextToken = null;
  assistantHistory = [];
  $("assistant-dock").hidden = true;
  $("assistant-panel").hidden = true;
  $("assistant-launcher").setAttribute("aria-expanded", "false");
}

function openAssistant() {
  if (!assistantContextToken) return;
  $("assistant-notice").hidden = true;
  $("assistant-panel").hidden = false;
  $("assistant-launcher").setAttribute("aria-expanded", "true");
  $("assistant-question").focus();
}

function closeAssistant() {
  $("assistant-panel").hidden = true;
  $("assistant-launcher").setAttribute("aria-expanded", "false");
  $("assistant-launcher").focus();
}

async function submitAssistantQuestion(event) {
  event.preventDefault();
  const questionInput = $("assistant-question");
  const question = questionInput.value.trim();
  if (!question || !assistantContextToken || $("assistant-send").disabled) return;

  questionInput.value = "";
  appendAssistantMessage("user", question);
  const history = assistantHistory.slice(-8);
  assistantHistory.push({ role: "user", content: question });
  assistantHistory = assistantHistory.slice(-8);
  const sendButton = $("assistant-send");
  sendButton.disabled = true;
  sendButton.textContent = "…";
  const pending = appendAssistantMessage("assistant", "Checking the incident context…");

  try {
    const response = await fetch(apiUrl("/api/assistant"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context_token: assistantContextToken, history }),
    });
    const body = await response.json();
    pending.remove();
    if (!response.ok || !body.success) {
      throw new Error(body.error || "The assistant could not answer this question.");
    }
    appendAssistantMessage("assistant", body.answer);
    assistantHistory.push({ role: "assistant", content: body.answer });
    assistantHistory = assistantHistory.slice(-8);
  } catch (error) {
    pending.remove();
    appendAssistantMessage("assistant", error.message || "Could not reach the incident assistant.");
  } finally {
    sendButton.disabled = !assistantContextToken;
    sendButton.textContent = "Send";
  }
}

// ---------- Actions ----------

async function analyze() {
  const text = input.value.trim();
  resetAssistant();
  $("analysis").hidden = true;
  $("severity-index").hidden = true;
  $("evidence-panel").hidden = true;
  $("warnings-panel").hidden = true;
  if (!text) {
    $("error-message").textContent = "Incident text is empty. Paste logs or an incident description first.";
    $("error-note").hidden = true;
    showState("error");
    return;
  }

  setBusy(true);
  showState("loading");
  try {
    const response = await fetch(apiUrl("/api/analyze"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    let body;
    try { body = await response.json(); }
    catch { body = { success: false, error: `Server returned an unexpected response (HTTP ${response.status}).` }; }

    if (body.success) {
      showState(null);
      renderSeverity(body.severity_assessment);
      renderAnalysis(body.analysis, body.extracted_signals);
      renderEvidence(body.similar_incidents || [], body.analysis.evidence_incidents || []);
      renderWarnings(body.analysis.warnings);
      if (body.assistant_context) {
        assistantContextToken = body.assistant_context;
        analyzedInputText = text;
        $("assistant-dock").hidden = false;
        $("assistant-notice").hidden = false;
        $("assistant-panel").hidden = true;
        $("assistant-launcher").setAttribute("aria-expanded", "false");
        appendAssistantMessage("assistant", "Ask a question about the current incident, its assessment, or retrieved historical records.");
      }
    } else {
      $("error-message").textContent = body.error || "The analysis failed for an unknown reason.";
      renderSeverity(body.severity_assessment);
      const evidence = body.similar_incidents || [];
      $("error-note").hidden = evidence.length === 0;
      showState("error");
      if (evidence.length) { renderEvidence(evidence, []); }
    }
  } catch (err) {
    $("error-message").textContent = `Could not reach the FastAPI backend at ${API_BASE || window.location.origin}. Start it with "uvicorn app.main:app --reload".`;
    $("error-note").hidden = true;
    showState("error");
  } finally {
    setBusy(false);
  }
}

async function loadFile(file) {
  const chip = $("file-chip");
  const showChip = (message, bad) => { chip.hidden = false; chip.textContent = message; chip.classList.toggle("bad", !!bad); };
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(extension)) { showChip(`Unsupported file type. Use ${ALLOWED_EXTENSIONS.join(", ")}`, true); return; }
  if (file.size > MAX_FILE_BYTES) { showChip("File too large (limit 2 MB)", true); return; }
  const content = await file.text();
  if (extension === ".json") {
    try { JSON.parse(content); } catch { showChip(`${file.name}: malformed JSON`, true); return; }
  }
  input.value = content;
  updateCharCount();
  invalidateAssistantContext();
  showChip(`Loaded ${file.name} (${(file.size / 1000).toFixed(1)} KB)`, false);
}

async function refreshStatus() {
  const dot = $("status-dot");
  const text = $("status-text");
  try {
    const response = await fetch(apiUrl("/health"));
    const health = await response.json();
    const s = health.services;
    const allUp = health.status === "healthy" && s.llm === "configured" && health.indexed_incidents > 0;
    dot.className = `dot ${allUp ? "dot-up" : "dot-degraded"}`;
    text.textContent = allUp ? "All systems operational" : "Degraded";
    const notes = [`API: ${s.api}`, `ChromaDB: ${s.chromadb} (${health.indexed_incidents} incidents)`,
      `Embedding model: ${s.embedding_model}`, `Reasoning service: ${s.llm}`];
    if (!health.indexed_incidents) notes.push("Run scripts/ingest_incidents.py");
    if (s.llm !== "configured") notes.push("Configure GROQ_API_KEY in .env");
    $("status").title = notes.join("\n");
  } catch {
    dot.className = "dot dot-down";
    text.textContent = "API unreachable";
  }
}

analyzeBtn.addEventListener("click", analyze);
$("assistant-form").addEventListener("submit", submitAssistantQuestion);
$("assistant-launcher").addEventListener("click", openAssistant);
$("assistant-notice-open").addEventListener("click", openAssistant);
$("assistant-close").addEventListener("click", closeAssistant);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("assistant-panel").hidden) closeAssistant();
});
$("sample-btn").addEventListener("click", async () => {
  const chip = $("file-chip");
  try {
    const response = await fetch(apiUrl("/sample"));
    if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
    input.value = await response.text();
    updateCharCount();
    invalidateAssistantContext();
    chip.hidden = false;
    chip.textContent = "Demo incident loaded";
    chip.classList.remove("bad");
  } catch {
    chip.hidden = false;
    chip.textContent = "Could not load demo; start the FastAPI backend first";
    chip.classList.add("bad");
  }
});
$("clear-btn").addEventListener("click", () => {
  input.value = ""; $("file-chip").hidden = true; updateCharCount();
  resetAssistant();
  $("analysis").hidden = true; $("severity-index").hidden = true;
  $("evidence-panel").hidden = true; $("warnings-panel").hidden = true;
  showState("empty");
});
$("file-input").addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) loadFile(file);
  event.target.value = ""; // allow re-selecting the same file
});
input.addEventListener("input", () => {
  updateCharCount();
  invalidateAssistantContext();
});
input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
});

updateCharCount();
refreshStatus();
setInterval(refreshStatus, 30000);

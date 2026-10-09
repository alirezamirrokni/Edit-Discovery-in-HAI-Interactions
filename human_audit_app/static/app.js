const state = {
  name: "",
  filename: "",
  samples: [],
  annotations: {},
  index: 0,
  saveCounter: 0,
  noteTimer: null,
};

const DATASETS = ["GPQA", "MATH-500", "MMLU-Pro", "PopQA"];
const el = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderText(value) {
  let text = escapeHtml(value || "");
  text = text.replace(/\*\*(.+?)\*\*/gs, "<strong>$1</strong>");
  text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  return text.replace(/\n/g, "<br>");
}

function initials(name) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "A";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function showToast(message, isError = false) {
  const toast = el("toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2400);
}

function setSaveState(mode, label) {
  const node = el("saveState");
  node.classList.remove("saved", "saving", "error");
  node.classList.add(mode);
  node.querySelector("span:last-child").textContent = label;
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let body = {};
  try { body = await response.json(); } catch (_) {}
  if (!response.ok || body.ok === false) {
    throw new Error(body.error || `Request failed (${response.status})`);
  }
  return body;
}

async function startSession() {
  const rawName = el("nameInput").value.trim();
  el("loginError").textContent = "";
  if (!rawName) {
    el("loginError").textContent = "Enter an annotator name to continue.";
    return;
  }
  const button = el("startButton");
  button.disabled = true;
  button.innerHTML = "Opening… <span>→</span>";
  try {
    const data = await api(`/api/session?name=${encodeURIComponent(rawName)}`);
    state.name = data.name;
    state.filename = data.filename;
    state.samples = data.samples;
    state.annotations = data.annotations || {};
    state.index = Number.isInteger(data.start_index) ? data.start_index : 0;

    el("annotatorName").textContent = state.name;
    el("annotatorFile").textContent = state.filename;
    el("avatarInitials").textContent = initials(state.name);
    el("downloadButton").href = `/api/export?name=${encodeURIComponent(state.name)}`;
    el("downloadButton").setAttribute("download", state.filename);

    el("loginView").classList.add("hidden");
    el("appView").classList.remove("hidden");
    renderAll();
    showToast(data.completed ? `Welcome back — ${data.completed} annotations restored.` : "Audit file created. Your progress will autosave.");
  } catch (error) {
    el("loginError").textContent = error.message;
  } finally {
    button.disabled = false;
    button.innerHTML = "Begin audit <span>→</span>";
  }
}

function switchUser() {
  if (state.noteTimer) clearTimeout(state.noteTimer);
  state.name = "";
  state.filename = "";
  state.samples = [];
  state.annotations = {};
  state.index = 0;
  el("appView").classList.add("hidden");
  el("loginView").classList.remove("hidden");
  el("nameInput").value = "";
  el("nameInput").focus();
}

function currentSample() {
  return state.samples[state.index];
}

function currentAnnotation() {
  const sample = currentSample();
  return sample ? (state.annotations[sample.audit_id] || null) : null;
}

function datasetStats(dataset) {
  const items = state.samples.filter(s => s.dataset === dataset);
  const done = items.filter(s => Boolean(state.annotations[s.audit_id])).length;
  return { done, total: items.length };
}

function updateProgress() {
  const completed = Object.keys(state.annotations).length;
  const total = state.samples.length || 200;
  const pct = total ? Math.round((completed / total) * 100) : 0;
  el("completedCount").textContent = completed;
  el("progressPercent").textContent = `${pct}%`;
  el("progressRing").style.setProperty("--progress", `${pct * 3.6}deg`);
  el("linearProgress").style.width = `${pct}%`;
  const remaining = total - completed;
  el("progressCaption").textContent = remaining === 0 ? "Audit complete" : `${remaining} example${remaining === 1 ? "" : "s"} remaining`;
}

function renderDatasetList() {
  const active = currentSample()?.dataset;
  const list = el("datasetList");
  list.innerHTML = DATASETS.map(ds => {
    const stat = datasetStats(ds);
    return `<button class="dataset-item ${ds === active ? "active" : ""}" data-dataset="${escapeHtml(ds)}">
      <span class="dataset-dot"></span>
      <span class="dataset-name">${escapeHtml(ds)}</span>
      <span class="dataset-progress">${stat.done}/${stat.total}</span>
    </button>`;
  }).join("");
  list.querySelectorAll(".dataset-item").forEach(button => {
    button.addEventListener("click", () => jumpToDataset(button.dataset.dataset));
  });
}

function jumpToDataset(dataset) {
  let index = state.samples.findIndex(s => s.dataset === dataset && !state.annotations[s.audit_id]);
  if (index < 0) index = state.samples.findIndex(s => s.dataset === dataset);
  if (index >= 0) goTo(index);
}

function renderSample() {
  const sample = currentSample();
  if (!sample) return;
  const ann = currentAnnotation();

  el("sampleIndex").textContent = `Example ${state.index + 1} of ${state.samples.length}`;
  el("datasetBadge").textContent = sample.dataset;
  el("auditId").textContent = sample.audit_id;
  el("questionText").innerHTML = renderText(sample.question);
  el("modelAnswer").innerHTML = renderText(sample.model_answer || "No model response recorded.");
  el("goldAnswer").innerHTML = renderText(sample.gold_answer || sample.gold_final || "No reference answer recorded.");

  const finalValue = (sample.gold_final || "").trim();
  const answerValue = (sample.gold_answer || "").trim();
  const showFinal = finalValue && finalValue !== answerValue;
  el("goldFinalWrap").classList.toggle("hidden", !showFinal);
  el("goldFinal").innerHTML = renderText(finalValue);

  const confirm = ann?.decision === "Confirm";
  const edit = ann?.decision === "Edit";
  el("confirmButton").classList.toggle("selected", confirm);
  el("editButton").classList.toggle("selected", edit);
  el("editDetails").classList.toggle("disabled", !edit);

  const aPill = el("currentA");
  aPill.className = "a-pill " + (confirm ? "confirm" : edit ? "edit" : "neutral");
  aPill.textContent = confirm ? "A = 0 · Confirm" : edit ? "A = 1 · Edit" : "Not labeled";

  el("reasonChips").querySelectorAll("button").forEach(button => {
    button.classList.toggle("selected", edit && button.dataset.reason === (ann?.reason_category || ""));
  });
  el("noteInput").value = ann?.annotation_note || "";

  el("prevButton").disabled = state.index <= 0;
  el("nextButton").disabled = state.index >= state.samples.length - 1;
  renderDatasetList();
  updateProgress();

  el("questionText").scrollTop = 0;
  el("modelAnswer").scrollTop = 0;
  el("goldAnswer").scrollTop = 0;
}

function renderAll() {
  renderSample();
  updateProgress();
}

function goTo(index) {
  if (state.noteTimer) {
    clearTimeout(state.noteTimer);
    state.noteTimer = null;
    saveNoteSnapshot();
  }
  state.index = Math.max(0, Math.min(index, state.samples.length - 1));
  renderSample();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function nextUnreviewed() {
  if (!state.samples.length) return;
  for (let offset = 1; offset <= state.samples.length; offset++) {
    const idx = (state.index + offset) % state.samples.length;
    if (!state.annotations[state.samples[idx].audit_id]) {
      goTo(idx);
      return;
    }
  }
  showToast("All 200 examples are labeled.");
}

function annotationSnapshot(sample, ann) {
  return {
    name: state.name,
    audit_id: sample.audit_id,
    decision: ann?.decision || null,
    reason_category: ann?.reason_category || "",
    annotation_note: ann?.annotation_note || "",
  };
}

async function persist(sample, ann) {
  const token = ++state.saveCounter;
  setSaveState("saving", "Saving…");
  try {
    const data = await api("/api/annotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotationSnapshot(sample, ann)),
      keepalive: true,
    });
    if (token === state.saveCounter) setSaveState("saved", "Saved");
    return data;
  } catch (error) {
    setSaveState("error", "Save failed");
    showToast(error.message, true);
    throw error;
  }
}

function chooseDecision(decision) {
  const sample = currentSample();
  if (!sample) return;
  const existing = state.annotations[sample.audit_id];
  const togglingOff = existing?.decision === decision;

  if (togglingOff) {
    delete state.annotations[sample.audit_id];
    renderSample();
    persist(sample, null).then(() => showToast("Label cleared."));
    return;
  }

  const ann = {
    decision,
    A: decision === "Edit" ? 1 : 0,
    reason_category: decision === "Confirm" ? "Correct as-is" : "",
    annotation_note: "",
  };
  state.annotations[sample.audit_id] = ann;
  renderSample();
  persist(sample, ann);
}

function chooseReason(reason) {
  const sample = currentSample();
  const ann = currentAnnotation();
  if (!sample || ann?.decision !== "Edit") return;
  ann.reason_category = ann.reason_category === reason ? "" : reason;
  state.annotations[sample.audit_id] = ann;
  renderSample();
  persist(sample, ann);
}

function scheduleNoteSave() {
  const sample = currentSample();
  const ann = currentAnnotation();
  if (!sample || ann?.decision !== "Edit") return;
  const auditId = sample.audit_id;
  const note = el("noteInput").value;
  ann.annotation_note = note;
  state.annotations[auditId] = ann;
  clearTimeout(state.noteTimer);
  setSaveState("saving", "Saving…");
  state.noteTimer = setTimeout(() => {
    const targetSample = state.samples.find(s => s.audit_id === auditId);
    const targetAnn = state.annotations[auditId];
    if (targetSample && targetAnn) persist(targetSample, targetAnn);
    state.noteTimer = null;
  }, 650);
}

function saveNoteSnapshot() {
  const sample = currentSample();
  const ann = currentAnnotation();
  if (!sample || ann?.decision !== "Edit") return;
  ann.annotation_note = el("noteInput").value;
  state.annotations[sample.audit_id] = ann;
  persist(sample, ann);
}

function onKeydown(event) {
  if (el("appView").classList.contains("hidden")) return;
  const tag = document.activeElement?.tagName?.toLowerCase();
  if (tag === "textarea" || tag === "input") return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;

  if (event.key.toLowerCase() === "c") chooseDecision("Confirm");
  else if (event.key.toLowerCase() === "e") chooseDecision("Edit");
  else if (event.key.toLowerCase() === "n") nextUnreviewed();
  else if (event.key === "ArrowLeft" && state.index > 0) goTo(state.index - 1);
  else if (event.key === "ArrowRight" && state.index < state.samples.length - 1) goTo(state.index + 1);
}

el("startButton").addEventListener("click", startSession);
el("nameInput").addEventListener("keydown", e => { if (e.key === "Enter") startSession(); });
el("switchUserButton").addEventListener("click", switchUser);
el("confirmButton").addEventListener("click", () => chooseDecision("Confirm"));
el("editButton").addEventListener("click", () => chooseDecision("Edit"));
el("reasonChips").querySelectorAll("button").forEach(button => button.addEventListener("click", () => chooseReason(button.dataset.reason)));
el("noteInput").addEventListener("input", scheduleNoteSave);
el("noteInput").addEventListener("blur", () => {
  if (state.noteTimer) {
    clearTimeout(state.noteTimer);
    state.noteTimer = null;
    saveNoteSnapshot();
  }
});
el("prevButton").addEventListener("click", () => goTo(state.index - 1));
el("nextButton").addEventListener("click", () => goTo(state.index + 1));
el("nextUnreviewedButton").addEventListener("click", nextUnreviewed);
document.addEventListener("keydown", onKeydown);
window.addEventListener("beforeunload", () => {
  if (state.noteTimer) saveNoteSnapshot();
});

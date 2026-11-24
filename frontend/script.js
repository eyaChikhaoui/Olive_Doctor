const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];

const state = { model: "leaf", file: null, previewUrl: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const modelLabels = { leaf: "Leaf Specialist", branch: "Branch Specialist" };

function setError(message = "") {
  const error = $("#errorMessage");
  error.textContent = message;
  error.hidden = !message;
}

function setModel(model) {
  state.model = model;
  $$(".specialist-card").forEach((card) => {
    const selected = card.dataset.model === model;
    card.classList.toggle("is-selected", selected);
    card.setAttribute("aria-checked", String(selected));
  });
  $("#selectedModelLabel").textContent = modelLabels[model];
  setError("");
}

function validateFile(file) {
  if (!file) return "Aucun fichier n’a été sélectionné.";
  const extension = `.${file.name.split(".").pop().toLowerCase()}`;
  if (!ACCEPTED_TYPES.includes(file.type) && !ACCEPTED_EXTENSIONS.includes(extension)) {
    return "Format non pris en charge. Utilisez une image JPG, PNG ou WebP.";
  }
  if (file.size > MAX_FILE_SIZE) return "L’image dépasse la taille maximale de 10 Mo.";
  if (file.size === 0) return "Le fichier importé est vide.";
  return "";
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

function setFile(file) {
  const validationError = validateFile(file);
  if (validationError) {
    clearFile();
    setError(validationError);
    return;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  $("#previewImage").src = state.previewUrl;
  $("#fileName").textContent = file.name;
  $("#fileSize").textContent = formatSize(file.size);
  $("#uploadEmpty").hidden = true;
  $("#uploadPreview").hidden = false;
  $("#analyzeButton").disabled = false;
  setError("");
}

function clearFile() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = null;
  state.previewUrl = null;
  $("#fileInput").value = "";
  $("#uploadEmpty").hidden = false;
  $("#uploadPreview").hidden = true;
  $("#analyzeButton").disabled = true;
}

function setLoading(isLoading) {
  $("#loadingState").hidden = !isLoading;
  $("#analyzeButton").disabled = isLoading || !state.file;
  $$(".specialist-card").forEach((card) => { card.disabled = isLoading; });
}

function renderResult(result) {
  $("#resultSpecialist").textContent = result.specialist_display_name;
  $("#predictedLabel").textContent = result.predicted_label_fr;
  $("#predictedClass").textContent = result.predicted_class;
  $("#confidenceValue").textContent = `${(result.confidence * 100).toFixed(1)} %`;
  $("#confidenceBar").style.width = `${result.confidence * 100}%`;
  $("#deviceNote").textContent = `Calcul effectué localement · ${result.device.toUpperCase()} · ${result.filename}`;
  const list = $("#probabilityList");
  list.replaceChildren();
  result.probabilities.forEach((item) => {
    const row = document.createElement("div");
    row.className = `probability-row${item.class_id === result.predicted_class ? " is-leading" : ""}`;
    row.innerHTML = `
      <span class="probability-label" title="${item.label_fr}">${item.label_fr}</span>
      <span class="probability-track"><span style="width:${item.probability * 100}%"></span></span>
      <span class="probability-value">${(item.probability * 100).toFixed(1)} %</span>
    `;
    list.appendChild(row);
  });
  $("#resultSection").hidden = false;
  $("#resultSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyze() {
  if (!state.file) return;
  setError("");
  setLoading(true);
  try {
    const form = new FormData();
    form.append("file", state.file);
    const response = await fetch(`/api/analyze/${state.model}`, { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "L’analyse n’a pas pu être terminée.");
    renderResult(payload);
  } catch (error) {
    setError(error.message || "Une erreur inattendue est survenue.");
  } finally {
    setLoading(false);
  }
}

function updateStatus(model, status) {
  const pill = $(`[data-status="${model}"]`);
  if (!pill) return;
  const ready = status.checkpoint_available && status.runtime_available;
  pill.textContent = ready ? "Disponible" : "À vérifier";
  pill.classList.toggle("is-ready", ready);
  pill.classList.toggle("is-error", !ready);
  pill.title = status.error || "Checkpoint et runtime disponibles";
}

async function loadModelStatuses() {
  try {
    const response = await fetch("/api/models/status");
    if (!response.ok) throw new Error("status");
    const payload = await response.json();
    payload.models.forEach((status) => updateStatus(status.key, status));
  } catch {
    ["leaf", "branch"].forEach((model) => {
      const pill = $(`[data-status="${model}"]`);
      if (pill) { pill.textContent = "Hors ligne"; pill.classList.add("is-error"); }
    });
  }
}

function bindUpload() {
  const dropzone = $("#dropzone");
  const input = $("#fileInput");
  $("#browseButton").addEventListener("click", (event) => { event.stopPropagation(); input.click(); });
  dropzone.addEventListener("click", (event) => {
    if (!event.target.closest("button")) input.click();
  });
  dropzone.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && !event.target.closest("button")) { event.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("is-dragging"); }));
  dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
  $("#removeButton").addEventListener("click", (event) => { event.stopPropagation(); clearFile(); setError(""); });
}

function bindPage() {
  $$(".specialist-card").forEach((card) => card.addEventListener("click", () => setModel(card.dataset.model)));
  $("#analyzeButton").addEventListener("click", analyze);
  $("#newAnalysisButton").addEventListener("click", () => {
    $("#resultSection").hidden = true;
    clearFile();
    setError("");
    $("#analyse").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  bindUpload();
  loadModelStatuses();
}

document.addEventListener("DOMContentLoaded", bindPage);

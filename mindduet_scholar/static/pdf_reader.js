import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.2.108/build/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.2.108/build/pdf.worker.mjs";

const root = document.querySelector(".pdf-reader");
const canvas = root.querySelector('[data-role="canvas"]');
const textLayer = root.querySelector('[data-role="text-layer"]');
const pageShell = root.querySelector('[data-role="page-shell"]');
const pageInput = root.querySelector('[data-role="page-input"]');
const pageCount = root.querySelector('[data-role="page-count"]');
const scaleSelect = root.querySelector('[data-role="scale-select"]');
const statusLine = root.querySelector('[data-role="status"]');
const selectedText = root.querySelector('[data-role="selected-text"]');
const contextPage = root.querySelector('[data-role="context-page"]');

let pdf = null;
let currentPage = 1;
let currentScale = Number(scaleSelect.value);
let currentPageText = "";
let rendering = false;
let pendingPage = null;

function setStatus(message) {
  statusLine.textContent = message;
}

function selectionInsideReader() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) {
    return "";
  }
  const range = selection.getRangeAt(0);
  if (!root.contains(range.commonAncestorContainer)) {
    return "";
  }
  return selection.toString().trim();
}

async function saveContext() {
  const selection = selectionInsideReader();
  if (selection) {
    selectedText.value = selection;
  }
  const response = await fetch(root.dataset.contextUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: Number(root.dataset.documentId),
      page: currentPage,
      scale: currentScale,
      selected_text: selectedText.value,
      page_text: currentPageText,
    }),
  });
  if (!response.ok) {
    setStatus("Context was not saved.");
    return;
  }
  setStatus(selectedText.value.trim() ? "Saved page and selection." : "Saved current page.");
}

function positionTextItem(span, item, viewport) {
  const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
  const fontHeight = Math.hypot(tx[2], tx[3]);
  span.style.left = `${tx[4]}px`;
  span.style.top = `${tx[5] - fontHeight}px`;
  span.style.fontSize = `${fontHeight}px`;
  span.style.transform = `scaleX(${item.width ? (item.width * viewport.scale) / Math.max(span.textContent.length * fontHeight * 0.52, 1) : 1})`;
}

async function renderPage(pageNumber) {
  if (rendering) {
    pendingPage = pageNumber;
    return;
  }
  rendering = true;
  const page = await pdf.getPage(pageNumber);
  const viewport = page.getViewport({ scale: currentScale });
  const context = canvas.getContext("2d");
  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  canvas.style.width = `${Math.floor(viewport.width)}px`;
  canvas.style.height = `${Math.floor(viewport.height)}px`;
  pageShell.style.width = `${Math.floor(viewport.width)}px`;
  pageShell.style.height = `${Math.floor(viewport.height)}px`;
  textLayer.replaceChildren();
  textLayer.style.width = `${Math.floor(viewport.width)}px`;
  textLayer.style.height = `${Math.floor(viewport.height)}px`;

  await page.render({ canvasContext: context, viewport }).promise;

  const text = await page.getTextContent();
  currentPageText = text.items.map((item) => item.str).join(" ").replace(/\s+/g, " ").trim();
  const fragment = document.createDocumentFragment();
  for (const item of text.items) {
    const span = document.createElement("span");
    span.textContent = item.str;
    span.setAttribute("role", "presentation");
    positionTextItem(span, item, viewport);
    fragment.appendChild(span);
  }
  textLayer.appendChild(fragment);

  currentPage = pageNumber;
  pageInput.value = String(currentPage);
  contextPage.textContent = String(currentPage);
  rendering = false;
  await saveContext();
  if (pendingPage !== null) {
    const next = pendingPage;
    pendingPage = null;
    await renderPage(next);
  }
}

function queuePage(pageNumber) {
  const bounded = Math.min(Math.max(1, pageNumber), pdf.numPages);
  renderPage(bounded).catch((error) => setStatus(`Render failed: ${error.message}`));
}

root.querySelector('[data-action="prev"]').addEventListener("click", () => queuePage(currentPage - 1));
root.querySelector('[data-action="next"]').addEventListener("click", () => queuePage(currentPage + 1));
root.querySelector('[data-action="save-selection"]').addEventListener("click", () => {
  saveContext().catch((error) => setStatus(`Save failed: ${error.message}`));
});

pageInput.addEventListener("change", () => queuePage(Number(pageInput.value)));
scaleSelect.addEventListener("change", () => {
  currentScale = Number(scaleSelect.value);
  queuePage(currentPage);
});

document.addEventListener("selectionchange", () => {
  const selection = selectionInsideReader();
  if (selection) {
    selectedText.value = selection;
  }
});

pdfjsLib.getDocument(root.dataset.pdfUrl).promise
  .then((loadedPdf) => {
    pdf = loadedPdf;
    pageCount.textContent = `/ ${pdf.numPages}`;
    pageInput.max = String(pdf.numPages);
    return renderPage(1);
  })
  .catch((error) => setStatus(`PDF load failed: ${error.message}`));

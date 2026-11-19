const API_BASE = "http://127.0.0.1:8000";
const fileInput = document.getElementById("fileInput");
const chooseBtn = document.getElementById("chooseBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const reportBtn = document.getElementById("reportBtn");
const preview = document.getElementById("preview");
const previewWrap = document.getElementById("previewWrap");
const fileInfo = document.getElementById("fileInfo");
const status = document.getElementById("status");
const resultCard = document.getElementById("resultCard");
const statusBadge = document.getElementById("statusBadge");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceLabel = document.getElementById("confidenceLabel");
const downloadBtn = document.getElementById("downloadBtn");
let selectedFile = null;
let lastReportData = null;

function resetView() {
  selectedFile = null;
  fileInput.value = "";
  preview.src = "";
  previewWrap.classList.add("hidden");
  fileInfo.textContent = "";
  analyzeBtn.disabled = true;
  resultCard.classList.add("hidden");
  status.textContent = "Ready for a new image.";
  statusBadge.textContent = "READY";
  statusBadge.style.background = "#e2e8f0";
  statusBadge.style.color = "#172033";
  confidenceFill.style.width = "0%";
  confidenceLabel.textContent = "0%";
  downloadBtn.href = "#";
  downloadBtn.classList.add("disabled");
  document.getElementById("resultSummary").textContent = "Waiting for analysis.";
  lastReportData = null;
}

async function loadImageDataUrl(imagePath) {
  if (!imagePath) return null;
  const url = imagePath.startsWith("http") ? imagePath : `${API_BASE}${imagePath}`;
  const response = await fetch(url);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function generateReport() {
  if (!lastReportData) return;

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: "mm", format: "a4" });

  const verdict = lastReportData.prediction || "AUTHENTIC";
  const tamperPercent = Number(lastReportData.probability_percent) || 0;
  const threshold = Number(lastReportData.threshold || 0);
  const regionCount = Array.isArray(lastReportData.regions) ? lastReportData.regions.length : 0;
  const note = lastReportData.localization_note || "No additional note provided.";

  doc.setFillColor(17, 32, 51);
  doc.rect(0, 0, 210, 32, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.text("TamperVision-Lite Forensic Report", 14, 20);

  doc.setTextColor(23, 32, 51);
  doc.setFontSize(11);
  doc.setFont("helvetica", "normal");
  doc.text(`Filename: ${lastReportData.filename}`, 14, 42);
  doc.text(`Verdict: ${verdict}`, 14, 50);
  doc.text(`Tampering probability: ${tamperPercent.toFixed(2)}%`, 14, 58);
  doc.text(`Decision threshold: ${threshold.toFixed(2)}`, 14, 66);
  doc.text(`Suspicious regions detected: ${regionCount}`, 14, 74);

  const summaryText =
    verdict === "TAMPERED"
      ? "The tampering detection model has classified this image as tampered. The probability score exceeds the configured threshold, and the localization map highlights suspicious regions where the editing pattern is most likely present. This indicates that the content has been modified, duplicated, pasted, or replaced in one or more areas of the image."
      : "The tampering detection model has classified this image as authentic. The probability score remains below the configured decision threshold, which means the visual content does not show significant evidence of localized editing or manipulation. The system therefore considers the image to be genuine within the model rules.";

  const interpretationText =
    "In this system, a tampering probability below 20% is treated as authentic, while a value at or above 20% is treated as tampered. This rule helps the user interpret the model output in simple terms: authentic images remain below the risk threshold, while tampered images exceed it and are highlighted visually.";

  const evidenceText =
    `The report is based on the model output, Grad-CAM heat map, pseudo-segmentation mask, and bounding box localization. The final result indicates ${tamperPercent.toFixed(2)}% tampering probability, which is ${verdict === "TAMPERED" ? "above" : "below"} the 20% decision rule. The generated localization overlays identify the suspicious region(s) and provide visual evidence of where anomalies were found.`;

  const noteLines = doc.splitTextToSize(note, 170);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.text("1. Executive Summary", 14, 92);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10.5);
  const summaryLines = doc.splitTextToSize(summaryText, 170);
  doc.text(summaryLines, 14, 100);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.text("2. Threshold Interpretation", 14, 130);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10.5);
  const interpretationLines = doc.splitTextToSize(interpretationText, 170);
  doc.text(interpretationLines, 14, 138);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.text("3. Evidence and Analysis", 14, 168);
  doc.setFont("helvetica", "normal");
  const evidenceLines = doc.splitTextToSize(evidenceText, 170);
  doc.text(evidenceLines, 14, 176);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.text("4. Localization Note", 14, 214);
  doc.setFont("helvetica", "normal");
  doc.text(noteLines, 14, 222);

  const imageEntries = [
    { key: "mask", label: "Pseudo-Mask" },
    { key: "localization", label: "Bounding Boxes" },
    { key: "gradcam", label: "Grad-CAM" }
  ];

  const loadedImages = [];
  for (const image of imageEntries) {
    const dataUrl = await loadImageDataUrl(lastReportData.images?.[image.key]);
    if (dataUrl) loadedImages.push({ ...image, dataUrl });
  }

  let imageX = 14;
  let imageY = 235;
  const imageWidth = 55;
  const imageHeight = 46;

  loadedImages.forEach((image, index) => {
    if (index > 0 && index % 3 === 0) {
      doc.addPage();
      imageX = 14;
      imageY = 20;
    }

    doc.setDrawColor(200, 210, 220);
    doc.roundedRect(imageX - 2, imageY - 4, imageWidth + 4, imageHeight + 14, 3, 3, "S");
    doc.addImage(image.dataUrl, "PNG", imageX, imageY, imageWidth, imageHeight);
    doc.setTextColor(23, 32, 51);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.text(image.label, imageX, imageY + imageHeight + 8);

    imageX += 60;
    if ((index + 1) % 3 === 0) {
      imageX = 14;
      imageY += 56;
    }
  });

  if (lastReportData.images?.final) {
    try {
      const finalUrl = await loadImageDataUrl(lastReportData.images.final);
      if (finalUrl) {
        doc.addPage();
        doc.setTextColor(23, 32, 51);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(16);
        doc.text("Final Visualization", 14, 18);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(10.5);
        doc.text("This final composite image combines the original content with the attention heat map and detected suspicious region overlay for quick visual inspection.", 14, 28, { maxWidth: 180 });
        doc.addImage(finalUrl, "PNG", 18, 38, 174, 120);
      }
    } catch (error) {
      console.warn("Final visualization could not be added to the PDF report.", error);
    }
  }

  doc.save(`tampervision_report_${Date.now()}.pdf`);
}

chooseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  if (!selectedFile) return;
  preview.src = URL.createObjectURL(selectedFile);
  previewWrap.classList.remove("hidden");
  fileInfo.textContent = `${selectedFile.name} · ${(selectedFile.size/1024/1024).toFixed(2)} MB`;
  analyzeBtn.disabled = false;
  status.textContent = "Ready to analyze.";
  resultCard.classList.add("hidden");
});

const dropzone = document.getElementById("dropzone");
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.style.borderColor = "#475569";
  dropzone.style.background = "#f8fafc";
});
dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "#cbd5e1";
  dropzone.style.background = "";
});
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.style.borderColor = "#cbd5e1";
  dropzone.style.background = "";
  const file = event.dataTransfer.files[0];
  if (!file) return;
  fileInput.files = event.dataTransfer.files;
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  fileInfo.textContent = `${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`;
  analyzeBtn.disabled = false;
  status.textContent = "Ready to analyze.";
  resultCard.classList.add("hidden");
});

resetBtn.addEventListener("click", resetView);
reportBtn.addEventListener("click", async () => {
  await generateReport();
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  status.textContent = "Analyzing image…";
  resultCard.classList.add("hidden");
  const form = new FormData();
  form.append("file", selectedFile);
  try {
    const response = await fetch(`${API_BASE}/predict`, {method:"POST", body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Prediction failed");

    lastReportData = {
      filename: data.filename,
      prediction: data.prediction,
      probability_percent: Number(data.probability_percent) || 0,
      threshold: Number(data.threshold) || 0,
      regions: Array.isArray(data.regions) ? data.regions : [],
      localization_note: data.localization_note || "",
      images: data.images || {}
    };

    const tamperedPercent = Number(data.probability_percent) || 0;
    const authenticPercent = 100 - tamperedPercent;
    const isTampered = tamperedPercent >= 20;
    const confidencePercent = isTampered ? tamperedPercent : authenticPercent;

    document.getElementById("prediction").textContent = isTampered ? "TAMPERED" : "AUTHENTIC";
    document.getElementById("prediction").className = isTampered ? "bad" : "ok";
    document.getElementById("probability").textContent = `${tamperedPercent.toFixed(2)}%`;
    document.getElementById("resultSummary").textContent = isTampered
      ? `Image is tampered. Tampering probability is ${tamperedPercent.toFixed(2)}%, which is above the 20% rule.`
      : `Image is authentic. Tampering probability is ${tamperedPercent.toFixed(2)}%, which is below the 20% rule.`;
    document.getElementById("threshold").textContent = data.threshold.toFixed(2);
    document.getElementById("regions").textContent = data.regions.length;
    document.getElementById("note").textContent = data.localization_note;
    document.getElementById("gradcam").src = `${API_BASE}${data.images.gradcam}`;
    document.getElementById("mask").src = `${API_BASE}${data.images.mask}`;
    document.getElementById("localization").src = `${API_BASE}${data.images.localization}`;
    document.getElementById("final").src = `${API_BASE}${data.images.final}`;
    downloadBtn.href = `${API_BASE}${data.images.final}`;
    downloadBtn.classList.remove("disabled");

    statusBadge.textContent = isTampered ? "TAMPERED" : "AUTHENTIC";
    statusBadge.style.background = isTampered ? "#fee2e2" : "#dcfce7";
    statusBadge.style.color = isTampered ? "#991b1b" : "#166534";
    confidenceFill.style.width = `${Math.max(0, Math.min(100, confidencePercent))}%`;
    confidenceLabel.textContent = `${confidencePercent.toFixed(1)}%`;

    resultCard.classList.remove("hidden");
    status.textContent = "Analysis complete.";
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
  }
});

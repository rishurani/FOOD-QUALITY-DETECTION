/**
 * FreshVision AI - Interactive Food Quality Analyzer
 * Handles drag-and-drop file upload, client-side validation, previewing,
 * AJAX submission to /api/predict, and rendering of model inference results.
 */

document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const uploadError = document.getElementById('uploadError');
    const uploadErrorText = document.getElementById('uploadErrorText');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingStage = document.getElementById('loadingStage');

    const guidanceCard = document.getElementById('guidanceCard');
    const resultCard = document.getElementById('resultCard');
    const modelMissingCard = document.getElementById('modelMissingCard');

    const resultBadge = document.getElementById('resultBadge');
    const resultConfidence = document.getElementById('resultConfidence');
    const confidenceLabel = document.getElementById('confidenceLabel');
    const confidenceBarFill = document.getElementById('confidenceBarFill');
    const resultExplanation = document.getElementById('resultExplanation');
    const metaTimestamp = document.getElementById('metaTimestamp');
    const metaModelVersion = document.getElementById('metaModelVersion');
    const metaRecordId = document.getElementById('metaRecordId');
    const metaCertainty = document.getElementById('metaCertainty');
    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

    let currentFile = null;

    // Supported file extensions and size limit (10MB)
    const ALLOWED_EXTS = ['jpg', 'jpeg', 'png', 'webp'];
    const MAX_BYTES = 10 * 1024 * 1024;

    // 1. Browse button triggers hidden input
    if (browseBtn && fileInput) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag and Drop listeners
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('drag-active');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('drag-active');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                handleFileSelection(files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFileSelection(fileInput.files[0]);
            }
        });
    }

    // 2. Validate and Preview File
    function handleFileSelection(file) {
        hideError();
        if (!file) return;

        // Check file extension
        const ext = file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTS.includes(ext)) {
            showError(`Invalid file format (.${ext}). Please select a JPG, JPEG, PNG, or WebP image.`);
            return;
        }

        // Check file size
        if (file.size > MAX_BYTES) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
            showError(`File is too large (${sizeMB} MB). Maximum allowable upload size is 10 MB.`);
            return;
        }

        currentFile = file;

        // Render preview via FileReader
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropzone.style.display = 'none';
            previewContainer.style.display = 'block';

            // Reset previous results
            resultCard.style.display = 'none';
            modelMissingCard.style.display = 'none';
            guidanceCard.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // 3. Clear file selection
    if (clearBtn) {
        clearBtn.addEventListener('click', resetAnalyzer);
    }
    if (analyzeAnotherBtn) {
        analyzeAnotherBtn.addEventListener('click', resetAnalyzer);
    }

    function resetAnalyzer() {
        currentFile = null;
        if (fileInput) fileInput.value = '';
        previewImage.src = '';
        previewContainer.style.display = 'none';
        dropzone.style.display = 'block';
        resultCard.style.display = 'none';
        modelMissingCard.style.display = 'none';
        guidanceCard.style.display = 'block';
        hideError();
    }

    // 4. Submit Image for Prediction
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            if (!currentFile) {
                showError('Please select or upload a valid food specimen image first.');
                return;
            }

            hideError();
            showLoading(true);

            const formData = new FormData();
            formData.append('image', currentFile);

            try {
                // Progress simulation for user feedback
                setTimeout(() => { if (loadingStage) loadingStage.textContent = 'Extracting Depthwise Convolutions...'; }, 300);
                setTimeout(() => { if (loadingStage) loadingStage.textContent = 'Calculating Freshness Probabilities...'; }, 700);

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.status === 503 || (data.model_loaded === false)) {
                    // Model is unavailable/not trained yet
                    showLoading(false);
                    guidanceCard.style.display = 'none';
                    resultCard.style.display = 'none';
                    modelMissingCard.style.display = 'block';
                    if (window.showToast) {
                        window.showToast('Model weights not found. Please run train_model.py', 'warning', 6000);
                    }
                    return;
                }

                if (!response.ok || !data.success) {
                    throw new Error(data.message || 'Image analysis failed.');
                }

                showLoading(false);
                displayPrediction(data);

                if (window.showToast) {
                    window.showToast('Food quality analysis complete!', 'success');
                }

            } catch (err) {
                showLoading(false);
                console.error('Prediction error:', err);
                showError(err.message || 'An error occurred while analyzing the image.');
                if (window.showToast) {
                    window.showToast(err.message || 'Analysis failed', 'error');
                }
            }
        });
    }

    // 5. Render Prediction Verdict
    function displayPrediction(data) {
        guidanceCard.style.display = 'none';
        modelMissingCard.style.display = 'none';
        resultCard.style.display = 'block';

        const confPct = (data.confidence * 100).toFixed(1);
        resultConfidence.textContent = `${confPct}%`;

        // Reset badge & gauge classes
        resultBadge.className = 'badge verdict-badge-large';
        confidenceBarFill.className = 'confidence-bar-fill';

        if (data.is_uncertain) {
            resultBadge.classList.add('badge-uncertain');
            resultBadge.textContent = 'Uncertain Quality';
            confidenceBarFill.classList.add('fill-uncertain');
            confidenceLabel.textContent = 'Borderline (< 65%)';
            metaCertainty.textContent = 'Uncertain';
            metaCertainty.style.color = 'var(--uncertain-color)';
        } else if (data.prediction === 'Fresh') {
            resultBadge.classList.add('badge-fresh');
            resultBadge.textContent = 'Fresh Quality';
            confidenceBarFill.classList.add('fill-fresh');
            confidenceLabel.textContent = 'High Visual Freshness';
            metaCertainty.textContent = 'Confident';
            metaCertainty.style.color = 'var(--primary)';
        } else {
            resultBadge.classList.add('badge-spoiled');
            resultBadge.textContent = 'Spoiled Quality';
            confidenceBarFill.classList.add('fill-spoiled');
            confidenceLabel.textContent = 'Visual Spoilage Detected';
            metaCertainty.textContent = 'Confident';
            metaCertainty.style.color = 'var(--spoiled-color)';
        }

        // Animate gauge bar width
        setTimeout(() => {
            confidenceBarFill.style.width = `${Math.min(100, Math.max(10, confPct))}%`;
        }, 50);

        resultExplanation.textContent = data.explanation || 'No detailed explanation generated.';
        metaTimestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        metaModelVersion.textContent = data.model_version || 'v1.0.0-MobileNetV2';
        metaRecordId.textContent = data.id ? `#${data.id}` : '--';
    }

    function showLoading(isLoading) {
        if (loadingOverlay) {
            loadingOverlay.style.display = isLoading ? 'flex' : 'none';
            if (loadingStage) loadingStage.textContent = 'Preprocessing Image Tensor...';
        }
        if (analyzeBtn) analyzeBtn.disabled = isLoading;
        if (clearBtn) clearBtn.disabled = isLoading;
    }

    function showError(msg) {
        if (uploadError && uploadErrorText) {
            uploadErrorText.textContent = msg;
            uploadError.style.display = 'flex';
        }
    }

    function hideError() {
        if (uploadError) {
            uploadError.style.display = 'none';
        }
    }
});

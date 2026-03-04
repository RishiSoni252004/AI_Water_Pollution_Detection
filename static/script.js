const dropArea = document.getElementById("drag-area");
const browseBtn = document.getElementById("browse-btn");
const fileInput = document.getElementById("file-input");
const detectBtn = document.getElementById("detect-btn");

const loader = document.getElementById("loader");
const resultSection = document.getElementById("result-section");

let selectedFile = null;

// Drag & Drop Functionality
browseBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", function() {
    selectedFile = this.files[0];
    showImagePreview();
});

dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("active");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("active");
});

dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    dropArea.classList.remove("active");
    selectedFile = event.dataTransfer.files[0];
    showImagePreview();
});

function showImagePreview() {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
        alert("Please upload a valid image file.");
        return;
    }

    const fileReader = new FileReader();
    fileReader.onload = () => {
        const fileURL = fileReader.result;
        // Update drag area to show image name or minimal preview
        dropArea.innerHTML = `<div class="icon"><i class="fas fa-image"></i></div>
                              <h3>${selectedFile.name}</h3>
                              <p style="color:var(--success); margin-top:10px;">Ready to analyze</p>`;
        
        // Setup the result preview image instantly
        document.getElementById("uploaded-image").src = fileURL;
        
        detectBtn.disabled = false;
    };
    fileReader.readAsDataURL(selectedFile);
}

// Prediction Process
detectBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Reset UI
    resultSection.classList.add("hidden");
    loader.classList.remove("hidden");
    detectBtn.disabled = true;

    // Form data
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
        } else {
            displayResults(data);
        }

    } catch (error) {
        console.error(error);
        alert("An error occurred while connecting to the server.");
    } finally {
        loader.classList.add("hidden");
        detectBtn.disabled = false;
        resultSection.classList.remove("hidden");
        
        // Scroll to results smoothly
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});

function displayResults(data) {
    const badge = document.getElementById("status-badge");
    const icon = document.getElementById("status-icon");
    const text = document.getElementById("prediction-text");
    const confidenceText = document.getElementById("confidence-score-text");
    const confidenceFill = document.getElementById("confidence-fill");
    const probsList = document.getElementById("probabilities-list");

    // Clean previous classes
    badge.className = "status-badge";
    icon.className = "";

    const pred = data.prediction.toLowerCase();
    text.textContent = predictionFormatting(pred);

    // Apply specific styles
    if (pred === 'clean') {
        badge.classList.add("status-clean");
        icon.classList.add("fas", "fa-check-circle");
    } else if (pred === 'algae') {
        badge.classList.add("status-warning");
        icon.classList.add("fas", "fa-exclamation-triangle");
    } else {
        // plastic, oil, general pollution
        badge.classList.add("status-polluted");
        icon.classList.add("fas", "fa-biohazard");
    }

    confidenceText.textContent = data.confidence;
    
    // Animate progress bar slightly delayed to allow DOM update
    setTimeout(() => {
        confidenceFill.style.width = data.confidence;
    }, 100);

    // Populate detailed probabilities
    probsList.innerHTML = "";
    for (const [cls, prob] of Object.entries(data.all_probabilities)) {
        const li = document.createElement("li");
        li.innerHTML = `<span>${predictionFormatting(cls)}</span> <span><strong>${prob}</strong></span>`;
        probsList.appendChild(li);
    }
}

function predictionFormatting(str) {
    // Capitalize and format
    const formatted = str.charAt(0).toUpperCase() + str.slice(1);
    return formatted.replace("_", " ");
}

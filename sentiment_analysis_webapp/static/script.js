function analyzeSentiment() {

    // Get the review text from the input field
    let reviewText = document.getElementById("reviewInput").value;

    // Prevent empty input submission
    if(!reviewText) {
        alert("Please enter a review to analyze");
        return;
    }

    // Send the review to backend API for prediction
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviews: [reviewText] })
    })
    .then(response => response.json())
    .then(data => {
        // Display prediction result
        if (data.sentiment_scores && data.sentiment_labels) {
            document.getElementById("result").innerHTML = 
                "Sentiment: " + data.sentiment_scores[0] + "<br>" +
                "Label: " + data.sentiment_labels[0];
        } else {
            document.getElementById("result").innerText = "Error: Unable to analyze sentiment";
        }
    })
    .catch(error => {
        console.error("Error:", error);
    });
}

function clearInput() {
    // Clear text input
    const reviewInput = document.getElementById("reviewInput");
    if (reviewInput) reviewInput.value = "";

    // Clear single review result
    const result = document.getElementById("result");
    if (result) result.innerText = "";

    // Clear upload result
    const uploadResult = document.getElementById("uploadResult");
    if (uploadResult) uploadResult.innerText = "";

    // For file input reset
    const fileInput = document.getElementById("fileInput");
    if (fileInput) fileInput.value = "";

    // Reset language detection icon
    const languageIcon = document.getElementById("languageIcon");
    if (languageIcon) languageIcon.innerText = "🌐";

    // Clear distribution summary
    const distributionSummary = document.getElementById("distributionSummary");
    if (distributionSummary) distributionSummary.innerText = "";
}

function goToUploadPage() {
    window.location.href = "/upload_page";
}

function goToHomePage() {
    window.location.href = "/";
}

// Function to upload file and analyze sentiment
function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    const uploadButton = document.querySelector(".btn-warning");
    const uploadResult = document.getElementById("uploadResult");
    const distributionSummary = document.getElementById("distributionSummary");

    if (!file) {
        alert("Please select a file to upload");
        return;
    }

    // Show loading spinner while processing
    uploadResult.innerHTML = `
        <strong>
            Processing... Please wait
            <span class="spinner-border text-primary spinner-border-sm ms-2" role="status" aria-hidden="true"></span>
        </strong>
    `;
    distributionSummary.innerText = "";
    uploadButton.disabled = true;

    // Prepare form data with CSV file
    const formData = new FormData();
    formData.append("file", file);

     // Send file to backend API
    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        uploadButton.disabled = false

        if (data.error) {
            console.error("Error:", data.error);
            uploadResult.innerText = data.error;
            return;
        }

         // Display overall sentiment result
        if (data.overall_sentiment && data.overall_score !== undefined) {
            const resultsHtml = `
                <p>Overall Sentiment Score: ${Number(data.overall_score).toFixed(2)}</p>
                <p>Label: ${data.overall_sentiment}</p>
             `;
            uploadResult.innerHTML = resultsHtml;

            // Display sentiment distribution (counts and percentages)
            if (data.distribution && data.distribution.percent && data.distribution.counts) {
                const d = data.distribution.percent;
                const c = data.distribution.counts;

                const summaryHTML = ` 
                    <p>Distribution Summary:</p>
                    <ul style="list-style-type: disc;">
                        <li>Positive: ${d.Positive}% (${c.Positive} reviews)</li>
                        <li>Neutral: ${d.Neutral}% (${c.Neutral} reviews)</li>
                        <li>Negative: ${d.Negative}% (${c.Negative} reviews)</li>
                    </ul>
                `;
                distributionSummary.innerHTML = summaryHTML;
            }
        } else {
            uploadResult.innerText = "No sentiment results found.";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        uploadButton.disabled = false;
        document.getElementById("uploadResult").innerText = "An error occurred while processing the file";
    });
}

// Function to detect language and update icon
function detectLanguage() {
    let text = document.getElementById("reviewInput").value.trim();

    // Reset icon if text is too short
    if (text.length < 3) {
        document.getElementById("languageIcon").innerText = "🌐";
        return;
    }

    // Send input to language detection API
    fetch("/detect_language", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.language) {
             // Emoji flags for known languages
            let languageIcons = {
                "en": "🇬🇧",
                "es": "🇪🇸",
                "fr": "🇫🇷",
                "de": "🇩🇪",
                "it": "🇮🇹",
                "pt": "🇵🇹",
                "nl": "🇳🇱",
                "pl": "🇵🇱",
                "ru": "🇷🇺",
                "ja": "🇯🇵",
                "ko": "🇰🇷",
                "zh": "🇨🇳",  
                "zh-cn": "🇨🇳",
                "zh-tw": "🇨🇳" 
            };

            const langKey = data.language.toLowerCase();
            const baseLang = langKey.split("-")[0]; // Handle cases like zh-cn
            const detectedIcon = languageIcons[data.language] || languageIcons[baseLang] || "🌐";

            document.getElementById("languageIcon").innerText = detectedIcon;

            console.log(` Detected Language: ${data.language} -> ${detectedIcon}`);
        }
    })
    .catch(error => console.error("Language detection error:", error));
}
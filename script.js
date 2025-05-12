async function predict() {
    const fileInput = document.getElementById('xray-upload');
    const resultsDiv = document.getElementById('results');
    
    if (!fileInput.files[0]) {
        alert('Please select an X-ray image first!');
        return;
    }

    resultsDiv.innerHTML = '<p>Analyzing... (This may take 10-20 seconds)</p>';
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        resultsDiv.innerHTML = `
            <div class="result-card">
                <h3>SVM Model</h3>
                <p>Grade: <span class="grade-${data.svm}">${data.svm}</span></p>
            </div>
            <div class="result-card">
                <h3>XGBoost Model</h3>
                <p>Grade: <span class="grade-${data.xgb}">${data.xgb}</span></p>
            </div>
            <div class="result-card">
                <h3>EfficientNet B6</h3>
                <p>Grade: <span class="grade-${data.effnet_b6}">${data.effnet_b6}</span></p>
            </div>
        `;
    } catch (error) {
        resultsDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}
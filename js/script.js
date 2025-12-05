document.addEventListener('DOMContentLoaded', function () {
    // Mobile Navigation Toggle
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.querySelector('.nav-links');

    if (hamburger) {
        hamburger.addEventListener('click', function () {
            this.classList.toggle('active');
            navLinks.classList.toggle('active');
        });
    }

    // Model Selection
    const modelOptions = document.querySelectorAll('.model-option');
    const selectedModelBanner = document.getElementById('selectedModelBanner');
    let selectedModel = 'primary';

    // Set up model selection
    modelOptions.forEach(option => {
        option.addEventListener('click', function () {
            // Update selected model
            selectedModel = this.getAttribute('data-model');

            // Update UI
            modelOptions.forEach(opt => opt.classList.remove('selected'));
            this.classList.add('selected');

            // Update banner
            const modelName = this.querySelector('h4').textContent;
            selectedModelBanner.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <span>Selected Model: <strong>${modelName}</strong></span>
            `;

            // Update the hidden radio input
            const radioInput = this.querySelector('input[type="radio"]');
            if (radioInput) {
                radioInput.checked = true;
            }
        });
    });

    // Initialize with first model selected
    if (modelOptions.length > 0) {
        modelOptions[0].click();
    }

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });

                // Close mobile menu if open
                if (navLinks.classList.contains('active')) {
                    hamburger.classList.remove('active');
                    navLinks.classList.remove('active');
                }
            }
        });
    });

    // File Upload Functionality
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressStatus = document.getElementById('progressStatus');
    const resultContainer = document.getElementById('resultContainer');
    const previewImage = document.getElementById('previewImage');
    const newAnalysisBtn = document.getElementById('newAnalysis');
    const modelSelect = document.getElementById('modelSelect');

    // Sample medical conditions and recommendations
    const conditions = {
        brain: [
            { name: 'Normal', confidence: 98.5 },
            { name: 'Glioma', confidence: 92.3 },
            { name: 'Meningioma', confidence: 88.7 },
            { name: 'Pituitary Tumor', confidence: 95.1 }
        ],
        lung: [
            { name: 'Normal', confidence: 96.2 },
            { name: 'Pneumonia', confidence: 91.8 },
            { name: 'COVID-19', confidence: 89.5 },
            { name: 'Tuberculosis', confidence: 87.3 }
        ],
        bone: [
            { name: 'Normal', confidence: 97.8 },
            { name: 'Fracture', confidence: 94.2 },
            { name: 'Osteoporosis', confidence: 88.6 },
            { name: 'Tumor', confidence: 82.4 }
        ]
    };

    let selectedFile = null;

    const recommendations = {
        'Normal': 'No significant abnormalities detected. Regular check-ups are recommended.',
        'Glioma': 'Consult with a neurosurgeon for further evaluation. MRI with contrast is recommended.',
        'Meningioma': 'Neurosurgical consultation is advised. Regular monitoring or surgical intervention may be necessary.',
        'Pituitary Tumor': 'Endocrine evaluation and MRI with contrast are recommended. Consultation with an endocrinologist and neurosurgeon is advised.',
        'Pneumonia': 'Antibiotic therapy is recommended. Chest X-ray follow-up in 4-6 weeks is advised.',
        'COVID-19': 'COVID-19 testing is recommended. Self-isolation and symptomatic treatment are advised.',
        'Tuberculosis': 'Sputum testing and chest CT are recommended. Infectious disease consultation is advised.',
        'Fracture': 'Orthopedic consultation is recommended. Immobilization and pain management may be necessary.',
        'Osteoporosis': 'Bone density scan and calcium/vitamin D supplementation are recommended.',
        'Tumor': 'Further imaging and biopsy are recommended. Oncology consultation is advised.'
    };

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('highlight');
    }

    function unhighlight() {
        dropZone.classList.remove('highlight');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFiles);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files } });
    }

    function handleFiles(e) {
        const files = e.target.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type.match('image.*')) {
                processFile(file);
            } else {
                alert('Please upload an image file (JPEG, PNG, etc.)');
            }
        }
    }

    function processFile(file) {
        selectedFile = file;
        // Show preview immediately
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            startAnalysis();
        };
        reader.readAsDataURL(file);
    }

    function startAnalysis() {
        // Hide upload area, show progress
        dropZone.style.display = 'none';
        progressContainer.style.display = 'block';
        updateProgress(10);
        progressStatus.textContent = 'Uploading image...';
        analyzeWithApi(selectedFile);
    }

    async function analyzeWithApi(file) {
        try {
            updateProgress(35);
            progressStatus.textContent = 'Processing and enhancing image...';

            const formData = new FormData();
            formData.append('file', file);
            const selectedModelInput = document.querySelector('input[name="model"]:checked');
            const selectedModel = selectedModelInput ? selectedModelInput.value : 'primary';
            formData.append('model', selectedModel);

            updateProgress(55);
            progressStatus.textContent = 'Running AI models...';

            let apiUrl = '/api/analyze';
            if (window.location.origin.startsWith('file:')) {
                apiUrl = 'http://127.0.0.1:8000/api/analyze';
            }
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const data = await response.json();

            updateProgress(85);
            progressStatus.textContent = 'Generating report...';

            // Populate results
            const cls = data.classification || {};
            const modelSel = data.selected_model || (document.getElementById('modelSelect') ? document.getElementById('modelSelect').value : 'primary');
            let bodyPart = 'Unknown';
            if (modelSel === 'primary') {
                bodyPart = cls.class_name || 'Unknown';
            } else if (modelSel === 'brain_tumor') {
                bodyPart = 'Brain';
            } else if (modelSel === 'fracture') {
                bodyPart = 'Bone';
            } else if (modelSel === 'lung_disease') {
                bodyPart = 'Chest';
            }
            document.getElementById('bodyPart').textContent = bodyPart;

            let conditionText = 'Analysis complete';
            let secondaryConf = null;
            let secondaryModelName = 'Secondary';

            if (modelSel === 'primary') {
                conditionText = cls.class_name ? `Detected: ${cls.class_name}` : 'Primary classification';
                if (data.segmentation && data.segmentation.performed) {
                    if (cls.class_name === 'Brain') {
                        conditionText += ' + Brain Tumor detection';
                        secondaryModelName = 'Brain Tumor';
                        if (data.segmentation && data.segmentation.labels && data.segmentation.labels.length > 0) {
                            const label = data.segmentation.labels[0];
                            conditionText = label.error ? 'Error: ' + label.error : `Detected: ${label.class}`;
                            secondaryConf = label.confidence;
                        } else {
                            conditionText = 'No detections';
                        }
                    } else if (cls.class_name === 'Bone') {
                        conditionText += ' + Fracture detection';
                        secondaryModelName = 'Fracture';
                        if (data.segmentation && data.segmentation.labels && data.segmentation.labels.length > 0) {
                            const label = data.segmentation.labels[0];
                            conditionText = label.error ? 'Error: ' + label.error : `Detected: ${label.class}`;
                            secondaryConf = label.confidence;
                        } else {
                            conditionText = 'No detections';
                        }
                    } else if (cls.class_name === 'Chest') {
                        conditionText += ' + Lung Disease detection';
                        secondaryModelName = 'Lung Disease';
                        if (data.segmentation && data.segmentation.labels && data.segmentation.labels.length > 0) {
                            const label = data.segmentation.labels[0];
                            conditionText = label.error ? 'Error: ' + label.error : `Detected: ${label.class}`;
                            secondaryConf = label.confidence;
                        } else {
                            conditionText = 'No detections';
                        }
                    } else {
                        conditionText += ' + Secondary analysis';
                    }
                }
            } else if (modelSel === 'brain_tumor') {
                if (data.segmentation && data.segmentation.labels && data.segmentation.labels.length > 0) {
                    const label = data.segmentation.labels[0];
                    conditionText = label.error ? 'Error: ' + label.error : `Detected: ${label.class}`;
                } else {
                    conditionText = 'No detections';
                }
            } else if (modelSel === 'fracture') {
                if (data.segmentation && data.segmentation.labels && data.segmentation.labels.length > 0) {
                    const label = data.segmentation.labels[0];
                    if (label.status) {
                        conditionText = `Status: ${label.status} (${label.num_fractures} fractures)`;
                    } else {
                        conditionText = label.error ? 'Error: ' + label.error : `Detected: ${label.class}`;
                    }
                } else {
                    conditionText = 'No fractures detected';
                }
            } else if (modelSel === 'lung_disease') {
                if (data.segmentation && data.segmentation.labels && data.segmentation.labels.length > 0) {
                    const label = data.segmentation.labels[0];
                    conditionText = label.error ? 'Error: ' + label.error : `Detected: ${label.class}`;
                } else {
                    conditionText = 'No lung disease detected';
                }
            }
            document.getElementById('condition').textContent = conditionText;

            let conf = '-';
            if (modelSel === 'primary') {
                const pConf = cls.confidence != null ? (cls.confidence * 100).toFixed(1) + '%' : '-';
                conf = `Primary: ${pConf}`;
                if (secondaryConf != null) {
                    conf += ` | ${secondaryModelName}: ${(secondaryConf * 100).toFixed(1)}%`;
                }
            } else {
                // For other models, get confidence from the first detection label
                if (data.segmentation && data.segmentation.labels && data.segmentation.labels.length > 0) {
                    const label = data.segmentation.labels[0];
                    if (label.confidence != null) {
                        conf = (label.confidence * 100).toFixed(1) + '%';
                    }
                }
            }
            document.getElementById('confidence').textContent = conf;
            document.getElementById('recommendations').textContent = data.notes || '—';

            if (data.annotated_image_base64) {
                previewImage.src = 'data:image/jpeg;base64,' + data.annotated_image_base64;
            }

            updateProgress(100);
            setTimeout(() => {
                showResults();

                // Setup Further Details button
                const furtherDetailsBtn = document.getElementById('furtherDetails');
                if (furtherDetailsBtn) {
                    // Remove old listeners to prevent duplicates if any
                    const newBtn = furtherDetailsBtn.cloneNode(true);
                    furtherDetailsBtn.parentNode.replaceChild(newBtn, furtherDetailsBtn);

                    newBtn.addEventListener('click', () => {
                        // Trigger Gemini Chat with diagnosis
                        if (window.chatBot) {
                            const prompt = `I have just analyzed a patient's scan using the AI model. 
                            Body Part: ${bodyPart}
                            Condition: ${conditionText}
                            Confidence: ${conf}
                            Notes: ${data.notes || 'None'}
                            
                            Please provide a detailed explanation of this condition, what it means for the patient, and recommend next steps or treatments.`;

                            // Open chat using the class method which handles state and focus
                            window.chatBot.toggleChat(true);

                            // Send message
                            window.chatBot.sendMessage(prompt);
                        }
                    });
                }
            }, 400);
        } catch (err) {
            console.error(err);
            updateProgress(100);
            document.getElementById('bodyPart').textContent = 'Error';
            document.getElementById('condition').textContent = 'Analysis failed';
            document.getElementById('confidence').textContent = '-';
            document.getElementById('recommendations').textContent = 'Could not contact the analysis server. Ensure server.py is running on http://127.0.0.1:8000.';
            setTimeout(showResults, 400);
        }
    }

    function updateProgress(progress) {
        progressBar.style.width = `${progress}%`;

        // Update steps based on progress
        const steps = document.querySelectorAll('.step');
        if (progress < 33) {
            steps[0].classList.add('active');
            steps[1].classList.remove('active');
            steps[2].classList.remove('active');
            progressStatus.textContent = 'Uploading and processing image...';
        } else if (progress < 66) {
            steps[0].classList.add('active');
            steps[1].classList.add('active');
            steps[2].classList.remove('active');
            progressStatus.textContent = 'Analyzing image with AI models...';
        } else {
            steps[0].classList.add('active');
            steps[1].classList.add('active');
            steps[2].classList.add('active');
            progressStatus.textContent = 'Generating diagnostic report...';
        }
    }

    function showResults() {
        progressContainer.style.display = 'none';
        resultContainer.style.display = 'block';
        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // New analysis button
    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', resetAnalysis);
    }

    function resetAnalysis() {
        // Reset file input
        fileInput.value = '';

        // Reset progress
        progressBar.style.width = '0%';
        progressStatus.textContent = 'Initializing analysis...';

        // Reset steps
        const steps = document.querySelectorAll('.step');
        steps[0].classList.add('active');
        steps[1].classList.remove('active');
        steps[2].classList.remove('active');

        // Reset results
        previewImage.src = '';
        document.getElementById('bodyPart').textContent = '-';
        document.getElementById('condition').textContent = '-';
        document.getElementById('confidence').textContent = '-';
        document.getElementById('recommendations').textContent = '-';

        // Show upload area, hide others
        dropZone.style.display = 'flex';
        progressContainer.style.display = 'none';
        resultContainer.style.display = 'none';

        // Scroll to upload section
        document.getElementById('diagnostic').scrollIntoView({ behavior: 'smooth' });
    }

    // AI Chat Widget logic is handled in chat.js

    // Contact Form
    const contactForm = document.getElementById('contactForm');
    const charCount = document.getElementById('charCount');
    const messageInput = document.getElementById('message');

    if (contactForm) {
        // Character count for message textarea
        messageInput.addEventListener('input', function () {
            const remaining = 500 - this.value.length;
            charCount.textContent = this.value.length;

            if (remaining < 0) {
                this.value = this.value.substring(0, 500);
                charCount.textContent = '500';
            }
        });

        // Form submission
        contactForm.addEventListener('submit', function (e) {
            e.preventDefault();

            // Get form data
            const formData = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                organization: document.getElementById('organization').value,
                message: messageInput.value
            };

            // Here you would typically send the data to a server
            console.log('Form submitted:', formData);

            // Show success message
            alert('Thank you for your message! We will get back to you soon.');

            // Reset form
            this.reset();
            charCount.textContent = '0';
        });
    }

    // Model cards click handlers
    const modelCards = document.querySelectorAll('.model-card .btn-outline');
    modelCards.forEach(card => {
        card.addEventListener('click', function () {
            const modelName = this.parentElement.querySelector('h3').textContent;
            alert(`You've selected the ${modelName}. This would open the specific model interface in a full implementation.`);

            // In a real implementation, this would load the specific model interface
            // For now, we'll just scroll to the diagnostic section
            document.getElementById('diagnostic').scrollIntoView({ behavior: 'smooth' });
        });
    });

    // Add active class to nav links on scroll
    const sections = document.querySelectorAll('section');
    const navItems = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', function () {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;

            if (pageYOffset >= sectionTop && pageYOffset < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${current}`) {
                item.classList.add('active');
            }
        });
    });

    // Initialize with home active
    navItems[0].classList.add('active');
});

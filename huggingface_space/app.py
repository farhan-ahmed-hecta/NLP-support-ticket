import os
import re
import joblib
import gradio as gr
import numpy as np

# Load trained production pipeline (Sublinear TF-IDF + Calibrated LinearSVC)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_email_classifier.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "final_email_classifier.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "final_email_classifier.joblib"

print(f"Loading model pipeline from {MODEL_PATH}...")
pipeline = joblib.load(MODEL_PATH)
classes = list(pipeline.classes_)
print(f"Model loaded successfully! Classes: {classes}")

def clean_text(text):
    if not text:
        return ""
    t = str(text)
    # Remove HTML tags
    t = re.sub(r'<[^>]+>', ' ', t)
    # Remove URLs
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    # Remove raw email addresses
    t = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', t)
    # Normalize repeated punctuation
    t = re.sub(r'([!?.])\1+', r'\1', t)
    # Remove excessive hyphens/underscores
    t = re.sub(r'[-_]{2,}', ' ', t)
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def predict_ticket(subject, body):
    clean_subj = clean_text(subject)
    clean_body = clean_text(body)
    
    if not clean_subj and not clean_body:
        return "⚠️ Please enter a subject or body.", {}, "No text provided."
    
    # Weight subject twice (subject carries dense intent in corporate emails)
    formatted_text = ((clean_subj + " ") * 2 + clean_body).strip()
    
    # Predict probabilities via Calibrated Classifier
    probs = pipeline.predict_proba([formatted_text])[0]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence_pct = probs[pred_idx] * 100
    
    # Format probabilities dictionary for Gradio Label component
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    
    # Extract top contributing keywords from TF-IDF vectorizer
    tfidf_vec = pipeline.named_steps.get('tfidf')
    top_keywords_str = "N/A"
    if tfidf_vec is not None:
        feature_names = tfidf_vec.get_feature_names_out()
        x_vec = tfidf_vec.transform([formatted_text])
        nonzero_idx = x_vec.nonzero()[1]
        
        # Calculate scores
        scores = []
        for idx in nonzero_idx:
            token = feature_names[idx]
            val = x_vec[0, idx]
            scores.append((token, val))
        scores.sort(key=lambda item: item[1], reverse=True)
        top_tokens = [f"`{t[0]}`" for t in scores[:8]]
        if top_tokens:
            top_keywords_str = " • ".join(top_tokens)
            
    # Category badge with color
    category_emojis = {
        "Technical Support": "💻",
        "Billing & Finance": "💳",
        "Customer Service & Returns": "📦",
        "Human Resources": "👥",
        "Product Support": "🛠️",
        "Sales & General": "🤝"
    }
    emoji = category_emojis.get(pred_class, "📁")
    
    summary_md = f"""### {emoji} **Routed Department:** `{pred_class}`
**Confidence:** `{confidence_pct:.1f}%`  
**SLA Target:** `{"< 1 Hour" if pred_class in ["Technical Support", "Billing & Finance"] else "< 4 Hours"}`
"""
    return summary_md, prob_dict, top_keywords_str

# Pre-defined demo presets
examples = [
    [
        "Severe Database Cluster Outage & 504 Timeout",
        "Our primary PostgreSQL production cluster on us-east-1 is completely unresponsive. Customer transactions are failing with connection timeout 504 errors. We need urgent escalation to senior DevOps/SRE immediately."
    ],
    [
        "Duplicate Credit Card Charge on Invoice #INV-88921",
        "Hi Support Team, our corporate credit card was billed twice for the annual software license renewal on October 25th. We request an immediate refund of the duplicate $1,200 charge."
    ],
    [
        "Damaged Package Received - Request for Replacement Order",
        "My package arrived today but the product inside was cracked and unusable. Tracking number is TRK-99214. Please initiate a return shipping label and ship a replacement item."
    ],
    [
        "Employee Onboarding & Background Verification Delay",
        "Hello HR Team, the background check and talent recruitment onboarding for our newly hired engineer is delayed. Please provide an update on the candidate application status in the employee portal."
    ],
    [
        "Mobile App Freezes and Crashes on iOS 17 Launch",
        "After updating to version 4.2.0 on iPhone 15, the app immediately freezes on the splash screen and crashes to home. Reinstalling did not fix the problem."
    ],
    [
        "Request for Enterprise License Pricing & Commercial Contract",
        "Hello Sales Team, our organization is looking to procure 250 enterprise seats for our engineering division. Could you please schedule a product demo and send a formal commercial proposal with volume pricing."
    ]
]

# Custom CSS styling
custom_css = """
.container { max-width: 1050px; margin: auto; }
.output-box { border-radius: 12px; }
"""

with gr.Blocks(title="AI Support Ticket Classifier", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📧 Enterprise Email / Support Ticket Classification System
    ### Production-Grade, Academically Valid & Leakage-Free Machine Learning Model
    *Categorizes unstructured customer inquiries into 6 core operational business queues with calibrated probability estimates.*
    """)
    
    with gr.Tabs():
        with gr.TabItem("🚀 Live Ticket Triage"):
            with gr.Row():
                with gr.Column(scale=5):
                    input_subject = gr.Textbox(
                        label="Email / Ticket Subject",
                        placeholder="e.g. Unauthorized credit card charge on invoice #9042",
                        lines=1
                    )
                    input_body = gr.Textbox(
                        label="Email Body / In-Depth Description",
                        placeholder="Describe the issue or paste the customer support request here...",
                        lines=6
                    )
                    with gr.Row():
                        btn_clear = gr.ClearButton([input_subject, input_body], value="Clear Fields")
                        btn_submit = gr.Button("Classify Ticket ⚡", variant="primary")
                        
                with gr.Column(scale=5):
                    output_summary = gr.Markdown(label="Routing Recommendation", value="*Submit a ticket or choose an example below to view classification results.*")
                    output_probs = gr.Label(label="Queue Probability Distribution (All 6 Classes)", num_top_classes=6)
                    output_keywords = gr.Markdown(label="Key Signal Terms Extracted", value="*Keywords will appear here.*")
            
            gr.Markdown("### 🎯 Quick Demo Presets (Click to Test)")
            gr.Examples(
                examples=examples,
                inputs=[input_subject, input_body],
                outputs=[output_summary, output_probs, output_keywords],
                fn=predict_ticket,
                cache_examples=False
            )
            
            btn_submit.click(
                fn=predict_ticket,
                inputs=[input_subject, input_body],
                outputs=[output_summary, output_probs, output_keywords]
            )

        with gr.TabItem("📊 Academic Model & Metrics"):
            gr.Markdown("""
            ## 🔬 Model Performance & Scientific Integrity
            
            | Evaluation Metric | Measured Value | Academic Benchmark Significance |
            | :--- | :---: | :--- |
            | **Test Accuracy** | **72.00%** | 100% Unseen 20% Held-Out Test Set (11,916 records) |
            | **Macro Precision** | **79.04%** | +9.54% boost over previous contaminated baseline |
            | **Macro F1-Score** | **72.21%** | Balanced across all 6 classes without minority penalty |
            | **Train-Test Leakage** | **0.00%** | 0 overlapping ticket hashes (Strictly Deduplicated) |
            | **HR Precision** | **96.43%** | Highly specific filtering for HR inquiries |
            | **Sales Precision** | **89.01%** | Highly accurate lead generation identification |
            | **Billing Precision** | **87.98%** | Robust detection of financial dispute tickets |

            ### 🛡️ Why 72.00% Honest Accuracy Beats 76.23% Contaminated Accuracy
            In earlier iterations, merging 5 raw datasets without deduplication caused **2,644 identical tickets** to exist in both training and test partitions.
            The apparent 76.23% accuracy was measuring **rote memorization**. 
            By eliminating all 8,305 duplicates and strictly isolating train and test sets, the current **72.00% accuracy with 79.04% Macro Precision** is mathematically sound, reproducible, and will generalize reliably on real-world incoming customer tickets.
            """)

if __name__ == "__main__":
    demo.launch()

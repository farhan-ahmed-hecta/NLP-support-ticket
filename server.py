import os
import sys
import json
import joblib
import numpy as np
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Setup model and knowledge base loading
print("Loading triage models and knowledge base...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

vectorizer = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.joblib'))
clf_queue = joblib.load(os.path.join(MODELS_DIR, 'clf_queue.joblib'))
clf_priority = joblib.load(os.path.join(MODELS_DIR, 'clf_priority.joblib'))
clf_type = joblib.load(os.path.join(MODELS_DIR, 'clf_type.joblib'))
nn_model = joblib.load(os.path.join(MODELS_DIR, 'nn_model.joblib'))

with open(os.path.join(MODELS_DIR, 'knowledge_base.json'), 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

with open(os.path.join(MODELS_DIR, 'queue_tag_map.json'), 'r', encoding='utf-8') as f:
    queue_tag_map = json.load(f)

with open(os.path.join(MODELS_DIR, 'sample_tickets.json'), 'r', encoding='utf-8') as f:
    sample_tickets = json.load(f)

with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r', encoding='utf-8') as f:
    metrics_data = json.load(f)

feature_names = vectorizer.get_feature_names_out()

if hasattr(clf_queue, 'coef_'):
    queue_coef = clf_queue.coef_
elif hasattr(clf_queue, 'calibrated_classifiers_'):
    queue_coef = np.mean([c.estimator.coef_ for c in clf_queue.calibrated_classifiers_], axis=0)
else:
    queue_coef = None

print("Models loaded successfully. Ready to serve inference.")

# Curated high-impact realistic demo presets aligned 1:1 with the 6 Core Categories
DEMO_PRESETS = [
    {
        "id": "preset_tech",
        "title": "💻 Tech: Database Crash & Outage",
        "category_expected": "Technical Support",
        "language": "en",
        "subject": "Severe Database Cluster Outage & 504 Timeout",
        "body": "Our primary PostgreSQL production cluster on us-east-1 is completely unresponsive. Customer transactions are failing with connection timeout 504 errors. We need urgent escalation to senior DevOps/SRE immediately."
    },
    {
        "id": "preset_billing",
        "title": "💳 Billing: Duplicate Invoice Charge",
        "category_expected": "Billing & Finance",
        "language": "en",
        "subject": "Duplicate Credit Card Charge on Invoice #INV-88921",
        "body": "Hi Support Team, our corporate credit card was billed twice for the annual software license renewal on October 25th. We request an immediate refund of the duplicate $1,200 charge."
    },
    {
        "id": "preset_returns",
        "title": "📦 Returns: Damaged Item Replacement",
        "category_expected": "Customer Service & Returns",
        "language": "en",
        "subject": "Damaged Package Received - Request for Replacement Order",
        "body": "My package arrived today but the product inside was cracked and unusable. Tracking number is TRK-99214. Please initiate a return shipping label and ship a replacement item."
    },
    {
        "id": "preset_hr",
        "title": "👥 HR: Employee Onboarding Delay",
        "category_expected": "Human Resources",
        "language": "en",
        "subject": "Employee Onboarding & Background Verification Delay",
        "body": "Hello HR Team, the background check and talent recruitment onboarding for our newly hired engineer is delayed. Please provide an update on the candidate application status in the employee portal."
    },
    {
        "id": "preset_product",
        "title": "🛠️ Product: Mobile App Freezing",
        "category_expected": "Product Support",
        "language": "en",
        "subject": "Mobile App Freezes and Crashes on iOS 17 Launch",
        "body": "After updating to version 4.2.0 on iPhone 15, the app immediately freezes on the splash screen and crashes to home. Reinstalling did not fix the problem."
    },
    {
        "id": "preset_sales",
        "title": "🤝 Sales: Enterprise Software License",
        "category_expected": "Sales & General",
        "language": "en",
        "subject": "Request for Enterprise License Pricing & Commercial Contract",
        "body": "Hello Sales Team, our organization is looking to procure 250 enterprise seats for our engineering division. Could you please schedule a product demo and send a formal commercial proposal with volume pricing."
    }
]

def detect_language(text):
    german_indicators = {'der', 'die', 'das', 'und', 'in', 'den', 'von', 'zu', 'mit', 'ist', 'im', 'fr', 'für', 'nicht', 'eine', 'einen', 'wir', 'bitte', 'dringend', 'fehler', 'problem', 'sehr', 'geehrte', 'unterstützung'}
    words = set(text.lower().split())
    overlap = words.intersection(german_indicators)
    return 'de' if len(overlap) >= 2 else 'en'

def run_triage(subject, body, specified_lang=None):
    clean_subj = subject.strip()
    clean_body = body.strip()
    full_text = (clean_subj + " ") * 3 + clean_body
    
    # Auto-detect language if not explicitly specified
    lang = specified_lang if specified_lang in ['en', 'de'] else detect_language(clean_subj + " " + clean_body)
    
    # Vectorize
    x = vectorizer.transform([full_text])
    matched_features = int(x.nnz)
    
    # 1. Queue Prediction & Probabilities
    q_probs = clf_queue.predict_proba(x)[0]
    q_top_idx = int(np.argmax(q_probs))
    pred_queue = str(clf_queue.classes_[q_top_idx])
    queue_confidence = float(q_probs[q_top_idx])
    
    print(f"[TRIAGE] Subject: '{clean_subj[:35]}...' | Queue: '{pred_queue}' ({queue_confidence*100:.1f}%) | Matched terms: {matched_features}")
    
    # Sort top 4 queues
    top_queue_indices = np.argsort(q_probs)[::-1][:4]
    queue_alternatives = []
    for idx in top_queue_indices:
        queue_alternatives.append({
            'queue': str(clf_queue.classes_[idx]),
            'probability': round(float(q_probs[idx]) * 100, 1)
        })
        
    # 2. Priority Prediction
    p_probs = clf_priority.predict_proba(x)[0]
    p_top_idx = int(np.argmax(p_probs))
    pred_priority = str(clf_priority.classes_[p_top_idx])
    priority_confidence = float(p_probs[p_top_idx])
    
    priority_breakdown = []
    for idx in np.argsort(p_probs)[::-1]:
        priority_breakdown.append({
            'priority': str(clf_priority.classes_[idx]),
            'probability': round(float(p_probs[idx]) * 100, 1)
        })
        
    sla_map = {
        'high': {'label': 'P1 Critical', 'target': '< 1 Hour SLA', 'color': 'red'},
        'medium': {'label': 'P2 Standard', 'target': '< 4 Hours SLA', 'color': 'amber'},
        'low': {'label': 'P3 Low Impact', 'target': '< 24 Hours SLA', 'color': 'emerald'}
    }
    sla_info = sla_map.get(pred_priority.lower(), {'label': 'Standard', 'target': '< 8 Hours SLA', 'color': 'blue'})

    # 3. Ticket Type Prediction
    t_probs = clf_type.predict_proba(x)[0]
    t_top_idx = int(np.argmax(t_probs))
    pred_type = str(clf_type.classes_[t_top_idx])
    type_confidence = float(t_probs[t_top_idx])

    # 4. Explainability: Top Contributing N-grams
    nonzero_indices = x.nonzero()[1]
    contributions = []
    if queue_coef is not None:
        for idx in nonzero_indices:
            val = x[0, idx]
            weight = float(queue_coef[q_top_idx, idx] * val)
            if weight > 0.05:
                contributions.append({
                    'token': str(feature_names[idx]),
                    'score': round(weight, 3)
                })
        contributions.sort(key=lambda k: k['score'], reverse=True)
    top_explain_tokens = contributions[:6]
    
    # 5. Suggested Tags
    # Candidate tags from queue map + extracted keywords
    suggested_tags = list(queue_tag_map.get(pred_queue, []))[:4]
    for item in top_explain_tokens[:2]:
        tok = item['token'].title()
        if tok not in suggested_tags and len(tok) > 2:
            suggested_tags.append(tok)
    suggested_tags = suggested_tags[:5]
    
    # 6. Nearest Neighbor Semantic Answer Retrieval
    distances, indices = nn_model.kneighbors(x, n_neighbors=2)
    top_match = knowledge_base[indices[0][0]]
    dist = float(distances[0][0])
    similarity = max(0.0, round((1.0 - dist) * 100, 1))
    
    # If the top match language differs from query, fallback to nearest matching language if available
    if top_match.get('language') != lang and len(indices[0]) > 1:
        alt_match = knowledge_base[indices[0][1]]
        if alt_match.get('language') == lang:
            top_match = alt_match
            similarity = max(0.0, round((1.0 - float(distances[0][1])) * 100, 1))
            
    suggested_response = {
        'answer': top_match['answer'],
        'reference_subject': top_match['subject'],
        'reference_queue': top_match['queue'],
        'match_confidence': similarity
    }
    
    return {
        'subject': clean_subj,
        'body': clean_body,
        'language': lang,
        'queue': {
            'predicted': pred_queue,
            'confidence': round(queue_confidence * 100, 1),
            'alternatives': queue_alternatives
        },
        'priority': {
            'predicted': pred_priority,
            'confidence': round(priority_confidence * 100, 1),
            'sla': sla_info,
            'breakdown': priority_breakdown
        },
        'type': {
            'predicted': pred_type,
            'confidence': round(type_confidence * 100, 1)
        },
        'tags': suggested_tags,
        'matched_features': matched_features,
        'explainability': top_explain_tokens,
        'suggested_response': suggested_response
    }

class TriageRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        # API Routes
        if path == '/api/stats':
            try:
                with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r', encoding='utf-8') as mf:
                    fresh_metrics = json.load(mf)
                self.send_json_response(fresh_metrics)
            except Exception:
                self.send_json_response(metrics_data)
            return
        elif path == '/api/presets':
            self.send_json_response(DEMO_PRESETS)
            return
        elif path == '/api/samples':
            self.send_json_response(sample_tickets[:40])
            return
        elif path == '/api/health':
            self.send_json_response({'status': 'healthy', 'models_loaded': True})
            return
            
        # Serve static assets
        if path == '/' or path == '':
            self.path = '/index.html'
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            body = json.loads(post_data) if post_data else {}
        except Exception as e:
            self.send_json_response({'error': f'Invalid JSON: {str(e)}'}, status=400)
            return

        if path == '/api/predict':
            subj = body.get('subject', '')
            txt_body = body.get('body', '')
            lang = body.get('language', None)
            if not subj and not txt_body:
                self.send_json_response({'error': 'Subject or body is required'}, status=400)
                return
            result = run_triage(subj, txt_body, lang)
            self.send_json_response(result)
            return
            
        elif path == '/api/batch':
            tickets = body.get('tickets', [])
            results = []
            for t in tickets[:50]: # cap batch to 50 for fast response
                subj = t.get('subject', '')
                txt_body = t.get('body', '')
                lang = t.get('language', None)
                res = run_triage(subj, txt_body, lang)
                results.append({
                    'id': t.get('id', ''),
                    'subject': subj,
                    'predicted_queue': res['queue']['predicted'],
                    'queue_confidence': res['queue']['confidence'],
                    'predicted_priority': res['priority']['predicted'],
                    'priority_confidence': res['priority']['confidence'],
                    'predicted_type': res['type']['predicted'],
                    'suggested_tags': res['tags'],
                    'response_draft_preview': res['suggested_response']['answer'][:140] + '...'
                })
            self.send_json_response({'count': len(results), 'results': results})
            return
            
        self.send_json_response({'error': 'Not Found'}, status=404)

    def send_json_response(self, data, status=200):
        response_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8080, open_browser=False):
    max_attempts = 10
    httpd = None
    selected_port = port
    
    for attempt in range(max_attempts):
        try:
            selected_port = port + attempt
            server_address = ('127.0.0.1', selected_port)
            httpd = HTTPServer(server_address, TriageRequestHandler)
            break
        except OSError:
            print(f"[INFO] Port {selected_port} is busy, trying port {selected_port + 1}...")
            
    if httpd is None:
        print(f"[ERROR] Could not bind to any port between {port} and {port + max_attempts - 1}.")
        return

    url = f"http://127.0.0.1:{selected_port}"
    print("=" * 65)
    print(f"🚀 AI Ticket Triage Server is LIVE at: {url}")
    print(f"   Also accessible at: http://localhost:{selected_port}")
    print("   Press Ctrl+C in this terminal to stop the server.")
    print("=" * 65)
    
    if open_browser:
        try:
            import webbrowser
            import threading
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
            
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server gracefully...")
        httpd.server_close()

if __name__ == '__main__':
    port = 8080
    auto_open = '--open' in sys.argv or '-o' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if args:
        try:
            port = int(args[0])
        except ValueError:
            pass
    run_server(port, open_browser=auto_open)

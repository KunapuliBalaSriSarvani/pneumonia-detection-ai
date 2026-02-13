<<<<<<< HEAD
# app.py

from flask import Flask, render_template, request, send_file, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import os
import datetime
from fpdf import FPDF
import json

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
HISTORY_FILE = "history.json"
MODEL_PATH = "pneumonia_model.pth"
# --- END CONFIGURATION ---

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load & Save History ---
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def save_history(history_data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f, indent=4)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_classes = 3
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model = model.to(device)

# ⚠️ Make sure this order matches your training dataset folder order
classes = ['BACTERIAL', 'NORMAL', 'VIRAL']

# --- Image Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Helper Functions ---
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    predicted_class = classes[predicted_idx.item()]

    # Debug: print prediction details
    print(f"[DEBUG] Image: {image_path} | Predicted: {predicted_class} | Confidence: {confidence.item()*100:.2f}%")
    print(f"[DEBUG] Class probabilities: {dict(zip(classes, probs.squeeze().tolist()))}")

    return predicted_class, confidence.item() * 100, probs.squeeze().tolist()

def infection_stage(prediction, confidence):
    if prediction != 'NORMAL':
        if confidence >= 75:
            return "Severe Infection"
        elif confidence >= 50:
            return "Moderate Infection"
        else:
            return "Mild Infection"
    return "Healthy"

def get_cause(prediction):
    if prediction == 'NORMAL':
        return "Lungs appear healthy. Keep maintaining a healthy lifestyle."
    elif prediction == 'BACTERIAL':
        return "Bacterial infection detected. Recommended to start antibiotics, maintain hydration, and consult a doctor."
    elif prediction == 'VIRAL':
        return "Viral infection detected. Recommended to rest, stay hydrated, and monitor oxygen levels if necessary."
    return "No information available."

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    context = {"history": list(reversed(load_history()))}
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            context["message"] = "No file was uploaded. Please select an image."
            return render_template("index.html", **context)

        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            prediction, confidence, probs = predict_image(file_path)
            stage = infection_stage(prediction, confidence)
            cause = get_cause(prediction)

            context.update({
                "uploaded_image": filename,
                "result": prediction,
                "confidence": round(confidence, 2),
                "stage": stage,
                "cause": cause
            })

            # Save to history
            history = load_history()
            history.append({
                "filename": filename,
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "stage": stage,
                "cause": cause,
                "probs": probs,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_history(history)

            context["history"] = list(reversed(history))

        except Exception as e:
            print(f"Error processing image: {e}")
            context["message"] = "An error occurred while processing the image."

    return render_template("index.html", **context)

@app.route("/history")
def view_history():
    return render_template("history.html", history=list(reversed(load_history())))

@app.route("/download_pdf/<int:record_index>")
def download_pdf(record_index):
    history = load_history()
    if record_index >= len(history):
        return "Record not found", 404

    record = history[record_index]
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "Chest X-Ray Diagnosis Report", ln=True, align="C")
    pdf.ln(5)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], record['filename'])
    if os.path.exists(img_path):
        pdf.image(img_path, x=55, w=90)

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 8, f"Disease: {record['prediction']}", ln=True)
    pdf.cell(0, 8, f"Stage: {record['stage']}", ln=True)
    pdf.cell(0, 8, f"Confidence: {record['confidence']}%", ln=True)
    pdf.cell(0, 8, f"Timestamp: {record['timestamp']}", ln=True)
    pdf.ln(5)

    safe_cause = record['cause'].replace('–', '-')  # Replace any special dash for PDF
    pdf.multi_cell(0, 7, f"Comment:\n{safe_cause}")

    pdf_file = f"{record['filename'].split('.')[0]}_report.pdf"
    pdf.output(pdf_file)  # No encoding argument to avoid UnicodeEncodeError
    return send_file(pdf_file, as_attachment=True)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return redirect(url_for('index'))

# --- Run ---
if __name__ == "__main__":
=======
# app.py

from flask import Flask, render_template, request, send_file, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import os
import datetime
from fpdf import FPDF
import json

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
HISTORY_FILE = "history.json"
MODEL_PATH = "pneumonia_model.pth"
# --- END CONFIGURATION ---

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load & Save History ---
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def save_history(history_data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f, indent=4)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_classes = 3
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model = model.to(device)

# ⚠️ Make sure this order matches your training dataset folder order
classes = ['BACTERIAL', 'NORMAL', 'VIRAL']

# --- Image Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Helper Functions ---
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    predicted_class = classes[predicted_idx.item()]

    # Debug: print prediction details
    print(f"[DEBUG] Image: {image_path} | Predicted: {predicted_class} | Confidence: {confidence.item()*100:.2f}%")
    print(f"[DEBUG] Class probabilities: {dict(zip(classes, probs.squeeze().tolist()))}")

    return predicted_class, confidence.item() * 100, probs.squeeze().tolist()

def infection_stage(prediction, confidence):
    if prediction != 'NORMAL':
        if confidence >= 75:
            return "Severe Infection"
        elif confidence >= 50:
            return "Moderate Infection"
        else:
            return "Mild Infection"
    return "Healthy"

def get_cause(prediction):
    if prediction == 'NORMAL':
        return "Lungs appear healthy. Keep maintaining a healthy lifestyle."
    elif prediction == 'BACTERIAL':
        return "Bacterial infection detected. Recommended to start antibiotics, maintain hydration, and consult a doctor."
    elif prediction == 'VIRAL':
        return "Viral infection detected. Recommended to rest, stay hydrated, and monitor oxygen levels if necessary."
    return "No information available."

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    context = {"history": list(reversed(load_history()))}
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            context["message"] = "No file was uploaded. Please select an image."
            return render_template("index.html", **context)

        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            prediction, confidence, probs = predict_image(file_path)
            stage = infection_stage(prediction, confidence)
            cause = get_cause(prediction)

            context.update({
                "uploaded_image": filename,
                "result": prediction,
                "confidence": round(confidence, 2),
                "stage": stage,
                "cause": cause
            })

            # Save to history
            history = load_history()
            history.append({
                "filename": filename,
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "stage": stage,
                "cause": cause,
                "probs": probs,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_history(history)

            context["history"] = list(reversed(history))

        except Exception as e:
            print(f"Error processing image: {e}")
            context["message"] = "An error occurred while processing the image."

    return render_template("index.html", **context)

@app.route("/history")
def view_history():
    return render_template("history.html", history=list(reversed(load_history())))

@app.route("/download_pdf/<int:record_index>")
def download_pdf(record_index):
    history = load_history()
    if record_index >= len(history):
        return "Record not found", 404

    record = history[record_index]
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "Chest X-Ray Diagnosis Report", ln=True, align="C")
    pdf.ln(5)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], record['filename'])
    if os.path.exists(img_path):
        pdf.image(img_path, x=55, w=90)

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 8, f"Disease: {record['prediction']}", ln=True)
    pdf.cell(0, 8, f"Stage: {record['stage']}", ln=True)
    pdf.cell(0, 8, f"Confidence: {record['confidence']}%", ln=True)
    pdf.cell(0, 8, f"Timestamp: {record['timestamp']}", ln=True)
    pdf.ln(5)

    safe_cause = record['cause'].replace('–', '-')  # Replace any special dash for PDF
    pdf.multi_cell(0, 7, f"Comment:\n{safe_cause}")

    pdf_file = f"{record['filename'].split('.')[0]}_report.pdf"
    pdf.output(pdf_file)  # No encoding argument to avoid UnicodeEncodeError
    return send_file(pdf_file, as_attachment=True)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return redirect(url_for('index'))

# --- Run ---
if __name__ == "__main__":
>>>>>>> 4328b919c88341e2918d5216bfaea5094334a6b4
    app.run(debug=True)
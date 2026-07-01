import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import httpx
import json


# ── Architecture (must match training exactly) ────────────────────────────────
class ConvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, use_skip=False):
        super().__init__()
        self.conv1    = ConvBNRelu(in_ch, out_ch)
        self.conv2    = ConvBNRelu(out_ch, out_ch)
        self.pool     = nn.MaxPool2d(2, 2)
        self.dropout  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_skip = use_skip
        if use_skip:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_skip:
            out = F.relu(out + self.skip(x), inplace=True)
        out = self.pool(out)
        return self.dropout(out)


class PotatoCNNFinal(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.b1 = Block(3,   32,  dropout=0.0,  use_skip=False)
        self.b2 = Block(32,  64,  dropout=0.0,  use_skip=False)
        self.b3 = Block(64,  128, dropout=0.1,  use_skip=True)
        self.b4 = Block(128, 256, dropout=0.1,  use_skip=True)
        self.b5 = Block(256, 512, dropout=0.15, use_skip=True)
        self.b6 = Block(512, 512, dropout=0.15, use_skip=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.b1(x); x = self.b2(x)
        x = self.b3(x); x = self.b4(x)
        x = self.b5(x); x = self.b6(x)
        x = self.gap(x)
        return self.classifier(x)


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = './models_scratch/potato_cnn_final_v5.pth'
IMAGE_SIZE    = 224
NUM_CLASSES   = 3
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/webp', 'image/jpg'}

# ── OpenRouter client config ──────────────────────────────────────────────────
# Add this in HuggingFace Spaces → Settings → Variables and Secrets:
#   OPENROUTER_API_KEY
OPENROUTER_MODEL_ID = 'openrouter/free'
OPENROUTER_API_KEY  = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_REFERER  = 'https://itzmoulik-potato-leaf-prediction-model.hf.space/'

# Disease info for frontend
DISEASE_INFO = {
    "Late Blight": {
        "severity": "severe",
        "description": """Late blight is a highly destructive disease of potato caused by the oomycete pathogen *Phytophthora infestans*. 
It primarily affects leaves, stems, and tubers. Initial symptoms appear as small, water-soaked lesions that rapidly expand into large, irregular, brown to black necrotic areas. 
Under humid conditions, a white, cottony fungal growth may be observed on the underside of leaves. 

The disease spreads aggressively in cool (10–20°C), wet, and humid environments, often leading to complete crop failure if not controlled. 
It was historically responsible for the Irish Potato Famine and remains one of the most economically significant potato diseases worldwide.""",

        "treatment": """Immediate intervention is critical. Apply protective and systemic fungicides such as mancozeb, chlorothalonil, or metalaxyl-based formulations. 
Ensure thorough coverage of foliage and repeat applications at recommended intervals during favorable disease conditions.

Remove and destroy infected plant material to reduce inoculum load. Avoid overhead irrigation and ensure proper drainage to minimize leaf wetness duration. 
In severe outbreaks, consider early harvesting of unaffected tubers to prevent tuber infection.

Always follow integrated pest management (IPM) practices and consult local agricultural guidelines for fungicide resistance management.""",

        "prevention": """Use certified disease-free seed tubers and plant resistant or tolerant potato varieties where available. 
Maintain adequate plant spacing to improve air circulation and reduce humidity within the canopy.

Avoid irrigation during late evening hours and prefer drip irrigation methods. Monitor weather conditions closely, as prolonged moisture and cool temperatures favor disease development. 
Implement crop rotation with non-host crops for at least 2–3 years.

Regular field scouting and early detection are crucial to prevent rapid disease spread."""
    },

    "Early Blight": {
        "severity": "moderate",
        "description": """Early blight is caused by the fungal pathogen *Alternaria solani* and commonly affects older foliage first. 
Symptoms include dark brown to black concentric lesions, often described as 'target spots'. Surrounding leaf tissue may turn yellow, leading to premature defoliation.

The disease thrives in warm (24–30°C), humid conditions and is commonly associated with plant stress such as nutrient deficiency or drought. 
Although less aggressive than late blight, severe infections can significantly reduce yield and tuber quality.""",

        "treatment": """Apply fungicides such as chlorothalonil or azoxystrobin at early stages of infection. 
Maintain a regular spray schedule, especially during favorable environmental conditions.

Remove infected leaves and ensure balanced fertilization, particularly adequate nitrogen levels. 
Reduce plant stress through proper irrigation and soil management practices.""",

        "prevention": """Practice crop rotation and avoid planting potatoes in the same field consecutively. 
Use disease-free seeds and maintain good field sanitation.

Ensure proper spacing between plants to reduce humidity and promote airflow. 
Regular monitoring and early intervention help in minimizing disease impact."""
    },

    "Healthy": {
        "severity": "none",
        "description": """The plant appears healthy with no visible signs of disease or stress. Leaves are green, well-formed, and free from lesions, discoloration, or deformities.

Healthy plants indicate proper nutrient balance, adequate water supply, and favorable environmental conditions.""",

        "treatment": """No treatment is required. Continue maintaining optimal growing conditions including proper irrigation, fertilization, and pest monitoring.""",

        "prevention": """Maintain good agricultural practices such as crop rotation, balanced fertilization, and regular monitoring for early signs of disease.

Ensure proper irrigation practices and avoid excessive moisture. Healthy crops are more resistant to diseases and environmental stress."""
    }
}

# ── System prompt builder ─────────────────────────────────────────────────────
def build_system_prompt(disease: str, confidence: float) -> str:
    info = DISEASE_INFO.get(disease, {})

    return f"""
ROLE:
You are an expert plant pathologist specializing in potato diseases, assisting farmers and researchers with AI-based leaf diagnosis.

CONTEXT:
The user has analyzed a potato leaf using an AI model.
- Disease: {disease}
- Confidence: {confidence:.1f}%
- Severity: {info.get('severity', 'unknown')}
- Description: {info.get('description', '')}
- Treatment: {info.get('treatment', '')}
- Prevention: {info.get('prevention', '')}

CORE BEHAVIOR:
- Provide practical, actionable advice that a farmer can directly apply.
- Keep answers clear and concise (2–4 sentences unless more detail is requested).
- Prioritize treatment, prevention, and next steps over theory.
- If confidence is low (<70%) or disease is "Uncertain", suggest retaking a clearer image and avoid strong conclusions.

DOMAIN GUARDRAILS:
- Focus on potato diseases, plant health, and agriculture-related topics.
- If the user slightly deviates (e.g., general farming, weather impact), still help if relevant.
- If the request is completely unrelated (coding, movies, etc.), respond politely:
  "I specialize in potato plant health and cannot assist with that topic."

CONVERSATION STYLE:
- Be helpful, calm, and practical — like an agricultural expert advising a farmer.
- Avoid overly technical jargon unless explicitly asked.
- Match the user's language (e.g., reply in Hindi if the user writes in Hindi).

SAFETY & CONSISTENCY:
- Do not contradict the diagnosis unless the user provides strong evidence.
- Do not fabricate unknown facts — if unsure, say so clearly.
- Do not provide harmful or unsafe agricultural practices.

GOAL:
Help the user understand the disease and take the correct next action to protect their crop.
"""


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str       # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str                        # latest user message
    disease_context: str                # e.g. "Early Blight"
    confidence: float                   # e.g. 94.5
    history: List[ChatMessage] = []     # previous turns for multi-turn chat


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title='Potato Disease Detector — v5')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory='static'), name='static')

# ── Load Model ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint           = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES          = checkpoint['class_names']
CONFIDENCE_THRESHOLD = checkpoint.get('confidence_threshold', 0.98)
TEMPERATURE          = checkpoint.get('temperature', 1.5)

MODEL = PotatoCNNFinal(NUM_CLASSES)
MODEL.load_state_dict(checkpoint['model_state_dict'])
MODEL.to(DEVICE)
MODEL.eval()

print(f'✅ Model loaded! (PotatoCNNFinal v5)')
print(f'   Device      : {DEVICE}')
print(f'   Classes     : {CLASS_NAMES}')
print(f'   Accuracy    : {checkpoint.get("test_accuracy", "N/A")}')
print(f'   Threshold   : {CONFIDENCE_THRESHOLD}')
print(f'   Temperature : {TEMPERATURE}')
print(f'   Chat model  : {OPENROUTER_MODEL_ID}')
print(f'   OpenRouter  : {"✅ set" if OPENROUTER_API_KEY else "❌ missing — set OPENROUTER_API_KEY secret"}')

# ── Transforms ────────────────────────────────────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_tta_images(img: Image.Image):
    """5 TTA augmentations."""
    return [
        img,
        TF.hflip(img),
        TF.vflip(img),
        TF.rotate(img, 90),
        TF.center_crop(img, int(IMAGE_SIZE * 0.9)),
    ]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get('/')
def index():
    return FileResponse('static/index.html')

@app.get('/health')
def health():
    return {
        'status'              : 'running',
        'model'               : 'PotatoCNNFinal v5 (Custom CNN from Scratch)',
        'device'              : str(DEVICE),
        'classes'             : CLASS_NAMES,
        'test_accuracy'       : checkpoint.get('test_accuracy'),
        'mean_confidence'     : checkpoint.get('mean_confidence'),
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'temperature'         : TEMPERATURE,
        'chat_model'          : OPENROUTER_MODEL_ID,
        'chat_ready'          : bool(OPENROUTER_API_KEY),
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Validate
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, 'Invalid file type. Use JPG, PNG or WebP.')
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, 'File too large. Max 10MB.')
    try:
        img = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        raise HTTPException(400, 'Could not read image.')

    # ── TTA inference ──
    tta_imgs  = get_tta_images(img)
    all_probs = []

    with torch.no_grad():
        for aug_img in tta_imgs:
            tensor = eval_transform(aug_img).unsqueeze(0).to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                out = MODEL(tensor)
            probs = torch.softmax(out.float() / TEMPERATURE, dim=1)[0]
            all_probs.append(probs)

    avg_probs       = torch.stack(all_probs).mean(dim=0)
    conf, pred_idx  = torch.max(avg_probs, dim=0)
    confidence      = conf.item()
    top2_vals, top2_idxs = torch.topk(avg_probs, 2)

    pred_class   = CLASS_NAMES[pred_idx.item()]
    is_uncertain = confidence < CONFIDENCE_THRESHOLD
    display_name = pred_class.replace('Potato___', '').replace('Potato__', '').replace('_', ' ').title()

    all_prob_dict = {
        CLASS_NAMES[i].replace('Potato___', '').replace('Potato__', '').replace('_', ' ').title():
            round(avg_probs[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    top2_list = [
        {
            'class': CLASS_NAMES[top2_idxs[i].item()].replace('Potato___', '').replace('Potato__', '').replace('_', ' ').title(),
            'confidence': round(top2_vals[i].item() * 100, 2)
        }
        for i in range(2)
    ]

    info_key = display_name if not is_uncertain else top2_list[0]['class']
    disease  = DISEASE_INFO.get(info_key, DISEASE_INFO['Healthy'])

    return {
        'predicted_class'    : display_name if not is_uncertain else 'Uncertain',
        'confidence'         : round(confidence * 100, 2),
        'is_uncertain'       : is_uncertain,
        'threshold'          : round(CONFIDENCE_THRESHOLD * 100, 2),
        'all_probabilities'  : all_prob_dict,
        'top2'               : top2_list,
        'disease_info'       : disease,
        'tta_passes'         : len(tta_imgs),
        'temperature'        : TEMPERATURE,
        'message'            : 'Low confidence — please upload a clearer image.' if is_uncertain else 'Prediction successful.',
    }


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post('/chat')
async def chat(req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, 'Chat unavailable — OPENROUTER_API_KEY secret not set in HF Spaces.')

    system_prompt = build_system_prompt(req.disease_context, req.confidence)

    # Build OpenAI-compatible messages list
    messages = [{"role": "system", "content": system_prompt}]
    for turn in req.history[-6:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": req.message})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": OPENROUTER_REFERER,
                    "X-Title": "Potato Leaf Disease Detector",
                },
                json={
                    "model": OPENROUTER_MODEL_ID,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            response.raise_for_status()
            data = response.json()
            reply = data['choices'][0]['message']['content']

        return {
            'reply'  : reply,
            'model'  : OPENROUTER_MODEL_ID,
            'disease': req.disease_context,
        }

    except httpx.HTTPStatusError as e:
        print(f"OpenRouter HTTP Error: {e.response.status_code} — {e.response.text}")
        raise HTTPException(500, f'OpenRouter error: {e.response.status_code} — {e.response.text}')
    except Exception as e:
        print(f"OpenRouter API Error: {str(e)}")
        raise HTTPException(500, f'OpenRouter error: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    # Hugging Face Spaces needs port 7860
    uvicorn.run(app, host='0.0.0.0', port=7860, reload=False)

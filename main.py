from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
import numpy as np
import joblib
import json
import os
from datetime import datetime
from create_model import create_and_save_model
import warnings
warnings.filterwarnings('ignore')

create_and_save_model()

print("Chargement du modele...")
model = tf.keras.models.load_model("models/model_universel.keras")
scaler = joblib.load("models/scaler.pkl")
le_type = joblib.load("models/le_type.pkl")
le_scenario = joblib.load("models/le_scenario.pkl")
with open("models/metadata.json") as f:
    metadata = json.load(f)
print("Pret!")

app = FastAPI(title="API Prediction Pannes Moteur", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

CATALOGUE = {
    "EL_S": "Electrique Petit",
    "EL_M": "Electrique Moyen",
    "EL_L": "Electrique Gros",
    "DI_S": "Diesel Petit",
    "DI_M": "Diesel Moyen",
    "DI_L": "Diesel Gros",
    "TURBOFAN": "Turbofan Avion",
    "HY_L": "Hydraulique Gros",
    "PO_L": "Pompe Grande",
    "CO_L": "Compresseur Gros",
}

SCENARIOS_NOM = {
    "NORMAL": "Normal",
    "SURCHAUFFE": "Surchauffe",
    "ROULEMENT": "Roulement use",
    "PRESSION_HUILE": "Pression huile",
    "SURCHARGE": "Surcharge",
    "ELECTRIQUE": "Defaut electrique",
    "DEGRADATION_HPC": "Degradation compresseur",
    "FUITE": "Fuite hydraulique",
    "CAVITATION": "Cavitation",
    "USURE_GENERALE": "Usure generale",
}

class DiagReq(BaseModel):
    type_moteur: str
    temperature: float
    pression: float
    puissance: float
    vibration: float
    presence: float = 1.0
    magnetique: float
    infrarouge: float
    machine_id: Optional[str] = None

@app.get("/")
async def accueil():
    return {
        "message": "API Prediction Pannes Moteur",
        "status": "en ligne",
        "version": "1.0.0",
        "documentation": "/docs",
        "types_moteurs": len(CATALOGUE),
        "scenarios": len(SCENARIOS_NOM),
    }

@app.get("/api/moteurs")
async def get_moteurs():
    moteurs = [{"code": k, "nom": v} for k, v in CATALOGUE.items()]
    return {"total": len(moteurs), "moteurs": moteurs}

@app.post("/api/diagnostic")
async def diagnostic(req: DiagReq):
    if req.type_moteur not in CATALOGUE:
        raise HTTPException(400, f"Type inconnu: {req.type_moteur}. Types valides: {list(CATALOGUE.keys())}")
    
    v = [req.temperature, req.pression, req.puissance, req.vibration, req.presence, req.magnetique, req.infrarouge]
    cn = scaler.transform([v]).astype(np.float32)
    te = le_type.transform([req.type_moteur])[0]
    pred = model.predict([cn, np.array([te], dtype=np.int32)], verbose=0)
    
    risque = round(float(pred[0][0][0]) * 100, 1)
    rul = max(0, int(float(pred[1][0][0]) * metadata["rul_max"]))
    anom = round(float(pred[2][0][0]) * 100, 1)
    si = int(np.argmax(pred[3][0]))
    sc = le_scenario.inverse_transform([si])[0]
    sp = round(float(pred[3][0][si]) * 100, 1)
    
    top3 = []
    for i in np.argsort(pred[3][0])[::-1][:3]:
        code = le_scenario.inverse_transform([int(i)])[0]
        prob = round(float(pred[3][0][i]) * 100, 1)
        top3.append({"code": code, "nom": SCENARIOS_NOM.get(code, code), "probabilite": prob})
    
    if risque > 80:
        urg = "CRITIQUE"
        action = "ARRET IMMEDIAT! Inspection obligatoire."
    elif risque > 50:
        urg = "ALERTE"
        action = "MAINTENANCE URGENTE dans les 24h."
    elif anom > 50:
        urg = "ATTENTION"
        action = "Valeurs anormales detectees. Verifier le moteur."
    else:
        urg = "NORMAL"
        action = "Tout va bien. Continuer la surveillance."
    
    return {
        "type_moteur": req.type_moteur,
        "nom_moteur": CATALOGUE[req.type_moteur],
        "machine_id": req.machine_id,
        "risque_panne": risque,
        "rul_cycles": rul,
        "score_anomalie": anom,
        "scenario": sc,
        "scenario_nom": SCENARIOS_NOM.get(sc, sc),
        "scenario_probabilite": sp,
        "top3_scenarios": top3,
        "niveau_urgence": urg,
        "action_recommandee": action,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/sante")
async def sante():
    return {
        "status": "en ligne",
        "modele_charge": True,
        "types_moteurs": len(CATALOGUE),
        "scenarios": len(SCENARIOS_NOM),
        "types_disponibles": list(CATALOGUE.keys()),
        "timestamp": datetime.now().isoformat(),
    }

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="API Prediction Pannes Moteur", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════
# PROFILS DES MOTEURS (valeurs normales)
# Au lieu de TensorFlow, on utilise un systeme de REGLES
# base sur les memes donnees que le modele IA
# ══════════════════════════════════════════════════════════════

PROFILS = {
    "EL_S": {"nom":"Electrique Petit","temp":(45,6),"pres":(1.0,0.1),"puis":(2.5,0.5),"vib":(0.5,0.1),"mag":(0.3,0.05),"ir":(38,5)},
    "EL_M": {"nom":"Electrique Moyen","temp":(60,8),"pres":(1.5,0.2),"puis":(25,5),"vib":(1.5,0.3),"mag":(0.6,0.08),"ir":(50,6)},
    "EL_L": {"nom":"Electrique Gros","temp":(85,12),"pres":(2.0,0.3),"puis":(200,40),"vib":(3.0,0.5),"mag":(1.2,0.15),"ir":(75,10)},
    "DI_S": {"nom":"Diesel Petit","temp":(80,8),"pres":(4.0,0.4),"puis":(25,5),"vib":(2.5,0.4),"mag":(0.01,0.005),"ir":(70,7)},
    "DI_M": {"nom":"Diesel Moyen","temp":(92,10),"pres":(4.5,0.5),"puis":(200,35),"vib":(3.5,0.6),"mag":(0.01,0.005),"ir":(82,8)},
    "DI_L": {"nom":"Diesel Gros","temp":(130,15),"pres":(7.0,1.0),"puis":(2000,300),"vib":(5.0,0.8),"mag":(0.01,0.005),"ir":(115,12)},
    "TURBOFAN": {"nom":"Turbofan Avion","temp":(500,30),"pres":(28,3),"puis":(22000,2000),"vib":(4.5,0.8),"mag":(2.0,0.3),"ir":(450,25)},
    "HY_L": {"nom":"Hydraulique Gros","temp":(55,6),"pres":(280,25),"puis":(150,30),"vib":(2.2,0.4),"mag":(0.01,0.005),"ir":(48,5)},
    "PO_L": {"nom":"Pompe Grande","temp":(52,5),"pres":(12,1.5),"puis":(120,20),"vib":(1.8,0.3),"mag":(0.45,0.08),"ir":(45,4)},
    "CO_L": {"nom":"Compresseur Gros","temp":(75,8),"pres":(18,2),"puis":(130,25),"vib":(3.0,0.5),"mag":(0.8,0.1),"ir":(65,7)},
}

SCENARIOS_PATTERNS = {
    "SURCHAUFFE": {"temp": 2.0, "pres": -1.0, "vib": 0.5, "ir": 2.0, "mag": 0, "puis": 0.3},
    "ROULEMENT": {"temp": 0.2, "pres": 0, "vib": 3.0, "ir": 0.2, "mag": -0.5, "puis": 0.3},
    "PRESSION_HUILE": {"temp": 1.5, "pres": -3.0, "vib": 1.0, "ir": 1.2, "mag": 0, "puis": -0.5},
    "SURCHARGE": {"temp": 2.0, "pres": 0.5, "vib": 2.0, "ir": 1.8, "mag": 0.3, "puis": 2.0},
    "ELECTRIQUE": {"temp": 1.0, "pres": 0, "vib": 0.8, "ir": 0.8, "mag": -2.5, "puis": -1.5},
    "DEGRADATION_HPC": {"temp": 1.5, "pres": -1.5, "vib": 0.8, "ir": 1.2, "mag": 0.3, "puis": -1.0},
    "FUITE": {"temp": 0.8, "pres": -2.5, "vib": 0.3, "ir": 0.5, "mag": 0, "puis": -1.0},
    "CAVITATION": {"temp": 0.5, "pres": -1.5, "vib": 3.5, "ir": 0.3, "mag": 0, "puis": -0.5},
    "USURE_GENERALE": {"temp": 0.5, "pres": -0.5, "vib": 0.5, "ir": 0.4, "mag": -0.2, "puis": -0.5},
}

SCENARIOS_NOM = {
    "NORMAL":"Normal","SURCHAUFFE":"Surchauffe","ROULEMENT":"Roulement use",
    "PRESSION_HUILE":"Pression huile","SURCHARGE":"Surcharge",
    "ELECTRIQUE":"Defaut electrique","DEGRADATION_HPC":"Degradation compresseur",
    "FUITE":"Fuite hydraulique","CAVITATION":"Cavitation","USURE_GENERALE":"Usure generale",
}

SCENARIOS_PAR_TYPE = {
    "EL_S": ["SURCHAUFFE","ROULEMENT","ELECTRIQUE"],
    "EL_M": ["SURCHAUFFE","ROULEMENT","ELECTRIQUE","SURCHARGE","USURE_GENERALE"],
    "EL_L": ["SURCHAUFFE","ROULEMENT","ELECTRIQUE","SURCHARGE","USURE_GENERALE"],
    "DI_S": ["SURCHAUFFE","PRESSION_HUILE"],
    "DI_M": ["SURCHAUFFE","PRESSION_HUILE","SURCHARGE","USURE_GENERALE"],
    "DI_L": ["SURCHAUFFE","PRESSION_HUILE","SURCHARGE","USURE_GENERALE"],
    "TURBOFAN": ["SURCHAUFFE","DEGRADATION_HPC"],
    "HY_L": ["FUITE"],
    "PO_L": ["ROULEMENT","CAVITATION","USURE_GENERALE"],
    "CO_L": ["SURCHAUFFE","ROULEMENT","USURE_GENERALE"],
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


def analyser_moteur(type_moteur, temp, pres, puis, vib, mag, ir):
    """
    Analyse intelligente basee sur les ecarts par rapport aux valeurs normales
    et identification du scenario le plus probable
    """
    profil = PROFILS[type_moteur]
    
    # Calculer les ecarts en nombre d'ecarts-types (z-scores)
    z_temp = (temp - profil["temp"][0]) / max(profil["temp"][1], 0.001)
    z_pres = (pres - profil["pres"][0]) / max(profil["pres"][1], 0.001)
    z_puis = (puis - profil["puis"][0]) / max(profil["puis"][1], 0.001)
    z_vib = (vib - profil["vib"][0]) / max(profil["vib"][1], 0.001)
    z_mag = (mag - profil["mag"][0]) / max(profil["mag"][1], 0.001)
    z_ir = (ir - profil["ir"][0]) / max(profil["ir"][1], 0.001)
    
    # Score d'anomalie global (moyenne des valeurs absolues des z-scores)
    z_scores = [z_temp, z_pres, z_puis, z_vib, z_mag, z_ir]
    anomalie_score = np.mean([abs(z) for z in z_scores])
    
    # Score anomalie en pourcentage (0-100)
    score_anomalie = min(100, max(0, anomalie_score * 25))
    
    # Identifier le scenario le plus probable
    scenarios_possibles = SCENARIOS_PAR_TYPE.get(type_moteur, [])
    
    meilleur_scenario = "NORMAL"
    meilleur_score = 0
    scores_scenarios = {}
    
    for scenario_id in scenarios_possibles:
        pattern = SCENARIOS_PATTERNS[scenario_id]
        
        # Calculer la similarite entre les z-scores et le pattern
        score = 0
        score += z_temp * pattern["temp"]
        score += z_pres * pattern["pres"]
        score += z_vib * pattern["vib"]
        score += z_ir * pattern["ir"]
        score += z_mag * pattern["mag"]
        score += z_puis * pattern["puis"]
        
        scores_scenarios[scenario_id] = max(0, score)
        
        if score > meilleur_score:
            meilleur_score = score
            meilleur_scenario = scenario_id
    
    # Si aucun scenario ne depasse le seuil -> normal
    if meilleur_score < 2.0:
        meilleur_scenario = "NORMAL"
    
    # Calculer les probabilites avec softmax
    all_scores = {"NORMAL": max(0, 5.0 - meilleur_score)}
    all_scores.update(scores_scenarios)
    
    total = sum(np.exp(min(s, 10)) for s in all_scores.values())
    probabilites = {}
    for k, v in all_scores.items():
        probabilites[k] = round(np.exp(min(v, 10)) / total * 100, 1)
    
    # Top 3 scenarios
    top3 = sorted(probabilites.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_list = [{"code": k, "nom": SCENARIOS_NOM.get(k, k), "probabilite": v} for k, v in top3]
    
    # Risque de panne
    if meilleur_scenario == "NORMAL":
        risque = min(15, score_anomalie * 2)
    else:
        risque = min(100, meilleur_score * 15 + score_anomalie * 5)
    risque = round(risque, 1)
    
    # RUL estime
    if risque < 20:
        rul = 400
    elif risque < 40:
        rul = 200
    elif risque < 60:
        rul = 80
    elif risque < 80:
        rul = 30
    else:
        rul = 5
    
    # Scenario probabilite
    scenario_prob = probabilites.get(meilleur_scenario, 0)
    
    return {
        "risque": risque,
        "rul": rul,
        "anomalie": round(score_anomalie, 1),
        "scenario": meilleur_scenario,
        "scenario_prob": scenario_prob,
        "top3": top3_list,
    }


@app.get("/")
async def accueil():
    return {
        "message": "API Prediction Pannes Moteur",
        "status": "en ligne",
        "version": "1.0.0",
        "documentation": "/docs",
        "types_moteurs": len(PROFILS),
        "scenarios": len(SCENARIOS_NOM),
    }


@app.get("/api/moteurs")
async def get_moteurs():
    moteurs = [{"code": k, "nom": v["nom"]} for k, v in PROFILS.items()]
    return {"total": len(moteurs), "moteurs": moteurs}


@app.post("/api/diagnostic")
async def diagnostic(req: DiagReq):
    if req.type_moteur not in PROFILS:
        raise HTTPException(400, f"Type inconnu: {req.type_moteur}. Types valides: {list(PROFILS.keys())}")
    
    result = analyser_moteur(
        req.type_moteur,
        req.temperature, req.pression, req.puissance,
        req.vibration, req.magnetique, req.infrarouge
    )
    
    if result["risque"] > 80:
        urg = "CRITIQUE"
        action = "ARRET IMMEDIAT! Inspection obligatoire."
    elif result["risque"] > 50:
        urg = "ALERTE"
        action = "MAINTENANCE URGENTE dans les 24h."
    elif result["anomalie"] > 50:
        urg = "ATTENTION"
        action = "Valeurs anormales detectees. Verifier le moteur."
    elif result["anomalie"] > 25:
        urg = "SURVEILLANCE"
        action = "A surveiller lors de la prochaine maintenance."
    else:
        urg = "NORMAL"
        action = "Tout va bien. Continuer la surveillance."
    
    return {
        "type_moteur": req.type_moteur,
        "nom_moteur": PROFILS[req.type_moteur]["nom"],
        "machine_id": req.machine_id,
        "risque_panne": result["risque"],
        "rul_cycles": result["rul"],
        "score_anomalie": result["anomalie"],
        "scenario": result["scenario"],
        "scenario_nom": SCENARIOS_NOM.get(result["scenario"], result["scenario"]),
        "scenario_probabilite": result["scenario_prob"],
        "top3_scenarios": result["top3"],
        "niveau_urgence": urg,
        "action_recommandee": action,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/sante")
async def sante():
    return {
        "status": "en ligne",
        "modele_charge": True,
        "methode": "analyse statistique avancee",
        "types_moteurs": len(PROFILS),
        "scenarios": len(SCENARIOS_NOM),
        "types_disponibles": list(PROFILS.keys()),
        "timestamp": datetime.now().isoformat(),
    }

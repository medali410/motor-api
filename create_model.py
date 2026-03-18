import os
import numpy as np
import pandas as pd
import json
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_and_save_model():
    if os.path.exists('models/model_universel.keras'):
        print("Modele existe deja!")
        return
    
    os.makedirs('models', exist_ok=True)
    print("Creation du modele...")
    np.random.seed(42)
    
    profils = {
        'EL_S':{'temp':(45,6),'pres':(1.0,0.1),'puis':(2.5,0.5),'vib':(0.5,0.1),'mag':(0.3,0.05),'ir':(38,5)},
        'EL_M':{'temp':(60,8),'pres':(1.5,0.2),'puis':(25,5),'vib':(1.5,0.3),'mag':(0.6,0.08),'ir':(50,6)},
        'EL_L':{'temp':(85,12),'pres':(2.0,0.3),'puis':(200,40),'vib':(3.0,0.5),'mag':(1.2,0.15),'ir':(75,10)},
        'DI_S':{'temp':(80,8),'pres':(4.0,0.4),'puis':(25,5),'vib':(2.5,0.4),'mag':(0.01,0.005),'ir':(70,7)},
        'DI_M':{'temp':(92,10),'pres':(4.5,0.5),'puis':(200,35),'vib':(3.5,0.6),'mag':(0.01,0.005),'ir':(82,8)},
        'DI_L':{'temp':(130,15),'pres':(7.0,1.0),'puis':(2000,300),'vib':(5.0,0.8),'mag':(0.01,0.005),'ir':(115,12)},
        'TURBOFAN':{'temp':(500,30),'pres':(28,3),'puis':(22000,2000),'vib':(4.5,0.8),'mag':(2.0,0.3),'ir':(450,25)},
        'HY_L':{'temp':(55,6),'pres':(280,25),'puis':(150,30),'vib':(2.2,0.4),'mag':(0.01,0.005),'ir':(48,5)},
        'PO_L':{'temp':(52,5),'pres':(12,1.5),'puis':(120,20),'vib':(1.8,0.3),'mag':(0.45,0.08),'ir':(45,4)},
        'CO_L':{'temp':(75,8),'pres':(18,2),'puis':(130,25),'vib':(3.0,0.5),'mag':(0.8,0.1),'ir':(65,7)},
    }
    
    scenarios = {
        'NORMAL':{'effets':{'temp':0,'pres':0,'puis':0,'vib':0,'mag':0,'ir':0},'applicable':list(profils.keys())},
        'SURCHAUFFE':{'effets':{'temp':0.5,'pres':-0.1,'puis':0.1,'vib':0.15,'mag':0,'ir':0.4},'applicable':['EL_S','EL_M','EL_L','DI_S','DI_M','DI_L','TURBOFAN','CO_L']},
        'ROULEMENT':{'effets':{'temp':0.05,'pres':0,'puis':0.1,'vib':0.8,'mag':-0.1,'ir':0.05},'applicable':['EL_S','EL_M','EL_L','PO_L','CO_L']},
        'PRESSION_HUILE':{'effets':{'temp':0.3,'pres':-0.6,'puis':-0.1,'vib':0.3,'mag':0,'ir':0.25},'applicable':['DI_S','DI_M','DI_L']},
        'SURCHARGE':{'effets':{'temp':0.4,'pres':0.1,'puis':0.5,'vib':0.4,'mag':0.1,'ir':0.35},'applicable':['EL_M','EL_L','DI_M','DI_L']},
        'ELECTRIQUE':{'effets':{'temp':0.2,'pres':0,'puis':-0.3,'vib':0.2,'mag':-0.5,'ir':0.15},'applicable':['EL_S','EL_M','EL_L']},
        'DEGRADATION_HPC':{'effets':{'temp':0.3,'pres':-0.3,'puis':-0.2,'vib':0.2,'mag':0.1,'ir':0.25},'applicable':['TURBOFAN']},
        'FUITE':{'effets':{'temp':0.15,'pres':-0.5,'puis':-0.2,'vib':0.1,'mag':0,'ir':0.1},'applicable':['HY_L']},
        'CAVITATION':{'effets':{'temp':0.1,'pres':-0.3,'puis':-0.1,'vib':0.7,'mag':0,'ir':0.08},'applicable':['PO_L']},
        'USURE_GENERALE':{'effets':{'temp':0.1,'pres':-0.1,'puis':-0.1,'vib':0.15,'mag':-0.05,'ir':0.08},'applicable':['EL_M','EL_L','DI_M','DI_L','CO_L','PO_L']},
    }
    
    all_data = []
    for type_m, prof in profils.items():
        for scen_id, scen in scenarios.items():
            if type_m not in scen['applicable']:
                continue
            n_machines = 15 if scen_id == 'NORMAL' else 8
            for machine in range(1, n_machines + 1):
                max_c = np.random.randint(150 if scen_id == 'NORMAL' else 100, 250 if scen_id == 'NORMAL' else 180)
                debut_deg = int(max_c * np.random.uniform(0.6, 0.8))
                for cycle in range(1, max_c + 1):
                    deg = 0.0 if (scen_id == 'NORMAL' or cycle < debut_deg) else ((cycle - debut_deg) / (max_c - debut_deg)) ** 1.5
                    eff = scen['effets']
                    row = {'type_moteur': type_m, 'scenario': scen_id, 'rul': max_c - cycle}
                    for cap, key in [('temperature','temp'),('pression','pres'),('puissance','puis'),('vibration','vib'),('magnetique','mag'),('infrarouge','ir')]:
                        m, s = prof[key]
                        row[cap] = max(0, np.random.normal(m*(1+deg*eff[key]), s*(1+deg*0.3)))
                    row['presence'] = 1.0
                    row['panne'] = 1 if (scen_id != 'NORMAL' and cycle > max_c * 0.95) else 0
                    row['anomalie'] = 1 if (scen_id != 'NORMAL' and cycle > debut_deg) else 0
                    all_data.append(row)
    
    df = pd.DataFrame(all_data)
    print(f"Donnees: {len(df)} mesures")
    
    capteur_cols = ['temperature','pression','puissance','vibration','presence','magnetique','infrarouge']
    le_type = LabelEncoder()
    df['type_enc'] = le_type.fit_transform(df['type_moteur'])
    le_scenario = LabelEncoder()
    df['scen_enc'] = le_scenario.fit_transform(df['scenario'])
    n_types = len(le_type.classes_)
    n_scenarios = len(le_scenario.classes_)
    
    scaler = StandardScaler()
    X_cap = scaler.fit_transform(df[capteur_cols].values).astype(np.float32)
    X_type = df['type_enc'].values.astype(np.int32)
    y_p = df['panne'].values.astype(np.float32)
    y_a = df['anomalie'].values.astype(np.float32)
    y_s = df['scen_enc'].values.astype(np.int32)
    rul_max = float(df['rul'].max())
    y_r = (df['rul'].values / rul_max).astype(np.float32)
    
    X1, X2, X3, X4, y1, y2, y3, y4, y5, y6, y7, y8 = train_test_split(
        X_cap, X_type, y_p, y_a, y_r, y_s, test_size=0.2, random_state=42)
    
    inp_c = layers.Input(shape=(7,))
    inp_t = layers.Input(shape=(1,))
    emb = layers.Flatten()(layers.Embedding(n_types, 8)(inp_t))
    m = layers.Concatenate()([inp_c, emb])
    x = layers.Dropout(0.3)(layers.BatchNormalization()(layers.Dense(256, activation='relu')(m)))
    x = layers.Dropout(0.3)(layers.BatchNormalization()(layers.Dense(128, activation='relu')(x)))
    x = layers.Dropout(0.2)(layers.BatchNormalization()(layers.Dense(64, activation='relu')(x)))
    bb = layers.Dense(32, activation='relu')(x)
    op = layers.Dense(1, activation='sigmoid', name='panne')(layers.Dense(16, activation='relu')(bb))
    or_ = layers.Dense(1, activation='sigmoid', name='rul')(layers.Dense(16, activation='relu')(bb))
    oa = layers.Dense(1, activation='sigmoid', name='anomalie')(layers.Dense(16, activation='relu')(bb))
    os_ = layers.Dense(n_scenarios, activation='softmax', name='scenario')(layers.Dense(32, activation='relu')(bb))
    
    model = Model(inputs=[inp_c, inp_t], outputs=[op, or_, oa, os_])
    model.compile(optimizer='adam',
        loss={'panne':'binary_crossentropy','rul':'mse','anomalie':'binary_crossentropy','scenario':'sparse_categorical_crossentropy'},
        loss_weights={'panne':1.0,'rul':0.5,'anomalie':0.8,'scenario':1.0},
        metrics={'panne':['accuracy'],'rul':['mae'],'anomalie':['accuracy'],'scenario':['accuracy']})
    
    print("Entrainement...")
    model.fit([X1, X3], {'panne':y1,'rul':y5,'anomalie':y3,'scenario':y7},
        epochs=30, batch_size=64, validation_split=0.2, verbose=0,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)])
    
    model.save('models/model_universel.keras')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_type, 'models/le_type.pkl')
    joblib.dump(le_scenario, 'models/le_scenario.pkl')
    json.dump({
        'n_types': n_types,
        'n_scenarios': n_scenarios,
        'rul_max': rul_max,
        'types': list(le_type.classes_),
        'scenarios': list(le_scenario.classes_),
        'capteurs': capteur_cols
    }, open('models/metadata.json', 'w'))
    
    print("Modele sauvegarde!")

if __name__ == "__main__":
    create_and_save_model()

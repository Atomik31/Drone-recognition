# Drone-recognition

Détection de personnes en temps réel (YOLO) sur le flux vidéo des DJI Goggles 3.

## Principe

- **Mac** : les Goggles 3 ne sont pas une webcam, l'app DJI **Goggles3 USB Live** en a l'accès exclusif.
  Le script capture donc directement la fenêtre de l'app (même masquée).
- **Windows** : l'app expose une caméra virtuelle, lue directement par OpenCV comme une webcam.

Dans les deux cas, YOLO analyse chaque image. L'affichage et la détection tournent dans deux threads
séparés pour rester fluides.

## Prérequis

Communs : Python 3, app **Goggles3 USB Live** installée, lunettes branchées en USB-C.

- **macOS** (Apple Silicon, GPU via MPS) : donner la permission *Enregistrement de l'écran* au terminal
  ou à l'éditeur qui lance le script (Réglages Système → Confidentialité et sécurité), puis le redémarrer.
- **Windows** : dans l'app, activer la caméra virtuelle (*Virtual camera*). Une carte NVIDIA (CUDA) est
  recommandée, sinon la détection tourne sur le CPU, nettement plus lent (préférer `yolo11s.pt` ou `yolo11n.pt`).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

1. Ouvrir l'app Goggles3 USB Live et cliquer sur **CONNECT** (panneau Settings fermé).
2. Lancer :
   ```bash
   python detect.py
   ```
3. Appuyer sur `q` dans la fenêtre « Drone » pour quitter.

## Réglages (en haut de `detect.py`)

| Variable | Rôle | Défaut |
|---|---|---|
| `YOLO("yolo11m.pt")` | Modèle (`yolo11s.pt` plus rapide, `yolo11m.pt` plus précis) | `yolo11m` |
| `CONF` | Seuil de certitude minimal des détections | `0.5` |
| `TOP` | Mac : hauteur (pt) du bandeau de l'app à ignorer | `82` |
| `CAMERA` | Windows : index de la caméra virtuelle (essayer 1, 2... si besoin) | `0` |
| `classes=[0]` | Classes détectées (0 = personne) | personnes |

## Limites

- L'essai gratuit de l'app coupe la session après 10 min : recliquer sur **CONNECT**.
- Windows : le chemin caméra virtuelle n'a pas pu être testé (développé sur Mac).
- Mac : la capture d'écran n'est pas synchronisée avec le flux (31 FPS) : de rares images peuvent être dupliquées ou sautées.
- Les modèles COCO détectent mal les personnes très petites vues du ciel : prévoir un modèle entraîné
  pour la vue aérienne (ex. VisDrone) pour les vols réels.

from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 18)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, "Documentation Technique : World Model Auto-Encodeur", border=False, ln=1, align="C")
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = PDF()
pdf.add_page()

# Configuration commune
pdf.set_font("helvetica", size=12)
pdf.set_text_color(0, 0, 0)
pdf.set_margins(20, 20, 20)

def section_title(title):
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, title, ln=1)
    pdf.set_font("helvetica", size=12)
    pdf.set_text_color(0, 0, 0)

def body_text(text):
    pdf.multi_cell(0, 8, text)
    pdf.ln(2)

section_title("1. Introduction")
body_text("Ce document detaille l'architecture du premier 'World Model' integre au simulateur 2D. "
          "L'objectif de cette Intelligence Artificielle est de developper une intuition spatiale "
          "a partir de capteurs partiels (le Lidar).")

section_title("2. Principe de Fonctionnement (Le Probleme)")
body_text("Le robot est equipe d'un capteur Lidar qui mesure la distance aux obstacles sur 360 degres. "
          "Ce signal est un simple tableau (vecteur 1D) de 360 nombres. A partir de ce signal limite, "
          "le modele doit reconstruire 'mentalement' une carte semantique en 2D (64x64 cellules) de son "
          "environnement local. L'IA doit donc apprendre a identifier que certaines distances correspondent "
          "a des murs, de l'herbe ou des arbres, et meme estimer l'agencement cache derriere les obstacles.")

section_title("3. Architecture du Reseau de Neurones")
body_text("Le reseau prend la forme d'un Auto-Encodeur (AutoEncoder) asymetrique, divise en deux parties :")

pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 8, "A. L'Encodeur Spatial (LidarEncoder)", ln=1)
pdf.set_font("helvetica", size=12)
body_text("L'encodeur est un Reseau Convolutif 1D (CNN 1D). Il recoit le tenseur de taille (Batch, 1, 360). "
          "Il applique 3 couches de convolution avec un 'stride' (pas) de 2. A chaque couche, la resolution spatiale "
          "est divisee par deux, tandis que le nombre de canaux (features) augmente : \n"
          "- Couche 1 : 360 -> 180 points (32 canaux)\n"
          "- Couche 2 : 180 -> 90 points (64 canaux)\n"
          "- Couche 3 : 90 -> 45 points (128 canaux)\n\n"
          "Les donnees sont ensuite aplaties et passees dans une couche lineaire (Dense) pour obtenir un "
          "'Embedding' final de dimension 128. C'est la representation latente.")

pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 8, "B. Le Decodeur Semantique (MapDecoder)", ln=1)
pdf.set_font("helvetica", size=12)
body_text("Le decodeur utilise des Convolutions Transposees 2D (Deconv2D). Il prend l'Embedding (vecteur de 128) "
          "et le projette mathematiquement sous la forme d'une minuscule image tres dense de 8x8 pixels avec "
          "128 canaux d'information. Ensuite, il 'decompresse' spatialement l'image par etapes : \n"
          "- Deconv 1 : 8x8 -> 16x16 (64 canaux)\n"
          "- Deconv 2 : 16x16 -> 32x32 (32 canaux)\n"
          "- Deconv 3 : 32x32 -> 64x64 (N canaux, ou N = nombre de terrains possibles)\n\n"
          "Le resultat final est une grille 2D d'equations (logits) representant les probabilites de chaque type de "
          "terrain a chaque endroit de la carte egocentrique.")

section_title("4. Algorithme d'Entrainement")
body_text("L'entrainement s'effectue en comparant la carte predite par le decodeur avec la 'Verite Terrain' "
          "(Ground Truth). La Verite Terrain est obtenue directement depuis le moteur du simulateur, qui "
          "extrait la carte locale autour du robot et la tourne algorithmiquement pour l'aligner avec le regard "
          "du robot (Carte Egocentrique).\n\n"
          "L'erreur est calculee en utilisant la fonction CrossEntropyLoss, parfaitement adaptee pour de la "
          "classification pixel par pixel. Le modele optimise ensuite ses poids de maniere retroactive par "
          "descente de gradient stochastique (Optimiseur Adam).")

pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "World_Model_Documentation.pdf")
pdf.output(pdf_path)
print(f"PDF generated at {pdf_path}")

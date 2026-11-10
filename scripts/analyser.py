"""Analyse d'un entraînement : résumé chiffré, alertes et tableau de bord.

Exemples :
    python scripts/analyser.py                                   # le dernier entraînement de runs/
    python scripts/analyser.py --dossier runs/essai_2026-11-09_18h04
    python scripts/analyser.py --lissage 5 --estimer 20000

Produit, dans <dossier>/analyse/ : tableau_de_bord.png et resume.md.
"""
import argparse
from pathlib import Path

from echecs_ma.utils.analyse import alertes, charger, dernier_entrainement, resume, resume_en_texte, tableau_de_bord

RACINE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Analyse le journal d'un entraînement.")
    parser.add_argument("--dossier", type=Path, help="Dossier de l'entraînement (par défaut : le plus récent de runs/)")
    parser.add_argument("--lissage", type=int, help="Nombre de lignes de la moyenne glissante (par défaut : automatique)")
    parser.add_argument("--estimer", type=int, help="Nombre de parties pour l'estimation de durée "
                                                    "(par défaut : entrainement.nb_parties de la config)")
    args = parser.parse_args()

    dossier = args.dossier or dernier_entrainement(RACINE / "runs")
    entrainement = charger(dossier)
    res = resume(entrainement, args.estimer)
    liste_alertes = alertes(entrainement)
    texte = resume_en_texte(entrainement, res, liste_alertes)

    sortie = Path(dossier) / "analyse"
    sortie.mkdir(exist_ok=True)
    (sortie / "resume.md").write_text(texte, encoding="utf-8")
    image = tableau_de_bord(entrainement, sortie / "tableau_de_bord.png", args.lissage)

    print(texte)
    print(f"Tableau de bord : {image}")
    print(f"Résumé : {sortie / 'resume.md'}")


if __name__ == "__main__":
    main()
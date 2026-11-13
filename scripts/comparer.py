"""Compare deux entraînements (par exemple avec et sans apprentissage) sur leur phase d'exploitation.

Exemples :
    python scripts/comparer.py runs/essai_2026-11-11_16h02 runs/essai_2026-11-11_15h54
    python scripts/comparer.py runs/avec_app runs/temoin --graine 1

Le premier dossier est l'essai, le second le témoin : les différences sont essai − témoin.
Écrit le tableau dans <essai>/analyse/comparaison_<témoin>.md et l'affiche.
"""
import argparse
from pathlib import Path

from echecs_ma.utils.analyse import charger, comparaison_en_texte, comparer


def main():
    parser = argparse.ArgumentParser(description="Compare deux entraînements sur leur phase d'exploitation.")
    parser.add_argument("essai", type=Path, help="Dossier de l'entraînement à évaluer")
    parser.add_argument("temoin", type=Path, help="Dossier de l'entraînement de référence")
    parser.add_argument("--graine", type=int, default=42, help="Graine du bootstrap (résultats reproductibles)")
    args = parser.parse_args()

    texte = comparaison_en_texte(comparer(charger(args.essai), charger(args.temoin), args.graine))

    sortie = args.essai / "analyse"
    sortie.mkdir(exist_ok=True)
    chemin = sortie / f"comparaison_{args.temoin.resolve().name}.md"
    chemin.write_text(texte, encoding="utf-8")
    print(texte)
    print(f"Comparaison : {chemin}")


if __name__ == "__main__":
    main()
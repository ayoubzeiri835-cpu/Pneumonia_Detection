from pathlib import Path
from owlready2 import *

ONTOLOGY_IRI = "http://pneumo.org/ontology.owl"


def create_ontology(output_path=None):
    onto = get_ontology(ONTOLOGY_IRI)

    with onto:
        class Organe(Thing):
            pass

        class Poumon(Organe):
            pass

        class PoumonDroit(Poumon):
            pass

        class PoumonGauche(Poumon):
            pass

        class EtatPulmonaire(Thing):
            pass

        class Normal(EtatPulmonaire):
            pass

        class Pneumonie(EtatPulmonaire):
            pass

        class SigneRadiologique(Thing):
            pass

        class Opacite(SigneRadiologique):
            pass

        class Infiltration(SigneRadiologique):
            pass

        class a_localisation(ObjectProperty):
            domain = [EtatPulmonaire]
            range = [Poumon]

        class a_signe(ObjectProperty):
            domain = [EtatPulmonaire]
            range = [SigneRadiologique]

        class indique(ObjectProperty):
            domain = [SigneRadiologique]
            range = [EtatPulmonaire]

        Opacite.indique = [Pneumonie]
        Infiltration.indique = [Pneumonie]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onto.save(file=str(output_path))

    return onto


def explain_prediction(pred, confidence, zone="inconnue"):
    if pred == 0:
        return (
            f"L'analyse est classée NORMAL "
            f"(confiance : {(1 - confidence) * 100:.1f}%). "
            "Dans l'ontologie, Normal appartient à EtatPulmonaire."
        )

    location = {
        "gauche": "poumon gauche",
        "droite": "poumon droit",
        "bilatérale": "les deux poumons",
    }.get(zone, "zone non précisée")

    return (
        f"L'analyse est classée PNEUMONIE "
        f"(confiance : {confidence * 100:.1f}%). "
        f"La heatmap Grad-CAM présente une activation dominante vers {location}. "
        "Dans l'ontologie, Pneumonie appartient à EtatPulmonaire et peut être "
        "associée à un signe radiologique tel que Opacite."
    )


if __name__ == "__main__":
    create_ontology("ontology/pneumo_ontology.owl")
    print("Ontology saved to ontology/pneumo_ontology.owl")

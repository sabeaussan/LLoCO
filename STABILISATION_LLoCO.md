# Journal de modifications --- Stabilisation LLoCO (IndustryOR)

------------------------------------------------------------------------

## 1) Objectif du travail

Stabiliser l'exécution batch :

    python3 main.py batch --dataset IndustryOR --all

En réduisant :

-   Les RUN_FAILED (NameError, SyntaxError, AttributeError)
-   Les NO_OBJECTIVE
-   Les erreurs dues à des variables inventées par le LLM
-   Les utilisations incorrectes de DataLoader

------------------------------------------------------------------------

## 2) Fichiers modifiés

Les fichiers suivants ont été modifiés ou étendus dans le cadre de la
stabilisation :

-   **llm_utils.py**
-   **code_utils.py**
-   **main.py**
-   (optionnel selon version) optimisation des prompts dans `prompts/`

------------------------------------------------------------------------

## 3) Modifications techniques détaillées

### 3.1 Gating anti-variables non définies (LLM)

**Fichier modifié :** `llm_utils.py`\
**Fonctions impactées :** - `_llm_step_with_gating` -
`_define_variables` - (recommandé aussi pour `_define_objective` et
`_define_constraints`)

#### Problème

Le LLM générait parfois des variables non définies :

    NameError: name 'd' is not defined

#### Solution

Ajout d'un mécanisme de validation automatique :

1.  Appel du LLM
2.  Extraction du code
3.  Détection des undefined names via `code_utils.find_undefined_names`
4.  Si nécessaire, nouvelle requête avec feedback explicite

#### Impact

Réduction massive des RUN_FAILED liés aux variables inexistantes.

------------------------------------------------------------------------

### 3.2 Interdiction de DataLoader si api_doc vide

**Fichier modifié :** `llm_utils.py`\
**Fonction modifiée :** `_define_variables`

Bloc ajouté :

    if not (api_doc or "").strip():
        code_hint += (
            "\nIMPORTANT:\n"
            "- DataLoader has NO usable API for this problem (api_doc is empty).\n"
            "- Do NOT use DataLoader at all.\n"
            "- Extract all required numeric data directly from the problem text and encode it as Python lists/dicts.\n"
        )

#### Problème

Le LLM utilisait DataLoader même sans API documentée → erreurs runtime.

#### Impact

Empêche les hallucinations DataLoader → forte stabilisation.

------------------------------------------------------------------------

### 3.3 Correction compatibilité typing Python

**Fichier modifié :** `code_utils.py`

#### Problème

Erreur :

    TypeError: unsupported operand type(s) for |: '_GenericAlias' and 'NoneType'

#### Cause

Utilisation de :

    List[str] | None

#### Solution

Remplacement par :

    Optional[List[str]]

#### Impact

Évite crash au démarrage du batch.

------------------------------------------------------------------------

### 3.4 Ajout / restauration de sanitize_python

**Fichier modifié :** `code_utils.py`\
**Utilisé dans :** `main.py`

#### Problème

Erreur :

    AttributeError: module 'code_utils' has no attribute 'sanitize_python'

#### Solution

Réintégration de la fonction `sanitize_python()` dans `code_utils.py`.

#### Impact

Pipeline complet sans crash intermédiaire.

------------------------------------------------------------------------

## 4) Résultat observé

-   Forte diminution des RUN_FAILED
-   Plus de problèmes exécutés jusqu'au solveur
-   Les échecs restants sont majoritairement des écarts d'objectif
    (modélisation)
-   Le système est désormais structurellement stable

------------------------------------------------------------------------

## 5) Travail restant (qualité modèle)

Les FAILED restants concernent :

-   Sens Min/Max incorrect
-   Contraintes manquantes
-   Extraction imprécise des coefficients
-   Modèles infeasible

Ces points relèvent de l'amélioration des prompts et de la qualité
d'extraction des données, non plus d'erreurs techniques du pipeline.

------------------------------------------------------------------------

## 6) Recommandations futures

-   Appliquer la règle api_doc vide aussi dans `_define_objective` et
    `_define_constraints`
-   Étendre le gating à toutes les étapes LLM
-   Ajouter validation automatique pré-exécution (`py_compile`)
-   Vérifier systématiquement le statut solver
-   Ajouter logs d'audit du code généré

------------------------------------------------------------------------

## 7) Conclusion

La version actuelle est désormais **techniquement stabilisée** :

-   Plus de crash structurel du pipeline
-   Moins d'erreurs runtime liées au LLM
-   Les erreurs restantes sont principalement liées à la qualité de
    modélisation

Le framework LLoCO est maintenant dans un état suffisamment robuste pour
itérations avancées et amélioration fine des prompts.
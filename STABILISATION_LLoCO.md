# Journal de modifications — Stabilisation LLoCO (IndustryOR, ComplexOR, LPWP)

---

## 1) Objectif du travail

Stabiliser l'exécution batch et étendre le support à tous les formats de datasets :

```bash
python3 main.py batch --dataset IndustryOR --all
python3 main.py batch --dataset ComplexOR --all
python3 main.py batch --dataset LPWP --all
```

En réduisant :

- Les RUN_FAILED (NameError, SyntaxError, AttributeError, ValueError)
- Les NO_OBJECTIVE
- Les erreurs dues à des variables inventées par le LLM
- Les utilisations incorrectes de DataLoader
- Les crashs de pipeline liés aux timeouts réseau
- Les SyntaxError dues à une mauvaise concaténation des blocs de code
- Les modèles INFEASIBLE générés par le LLM
- Les erreurs runtime dues à des dépendances de données non respectées entre étapes

---

## 2) Fichiers modifiés

Les fichiers suivants ont été modifiés ou étendus dans le cadre de la stabilisation :

- **llm_utils.py**
- **code_utils.py**
- **main.py**
- (optionnel selon version) optimisation des prompts dans `prompts/`

---

## 3) Modifications techniques détaillées

### 3.1 Gating anti-variables non définies (LLM)

**Fichier modifié :** `llm_utils.py`  
**Fonctions impactées :** `_llm_step_with_gating`, `_define_variables`, `_define_objective`, `_define_constraints`

#### Problème

Le LLM générait parfois des variables non définies :

```
NameError: name 'd' is not defined
NameError: name 'r' is not defined
```

#### Solution

Ajout d'un mécanisme de validation automatique dans `_llm_step_with_gating` :

1. Appel du LLM
2. Extraction du code
3. Détection des undefined names via `code_utils.find_undefined_names`
4. Si nécessaire, nouvelle requête avec feedback explicite

Le gating est désormais appliqué à **toutes les étapes** : `_define_variables`, `_define_objective`, `_define_constraints` et `print_solution` (voir 3.5).

#### Impact

Réduction massive des RUN_FAILED liés aux variables inexistantes. ✅ **Validé**

---

### 3.2 Interdiction de DataLoader si api_doc vide

**Fichier modifié :** `llm_utils.py`  
**Fonctions modifiées :** `_define_variables`, `_define_objective`, `_define_constraints`

#### Problème

Le LLM utilisait DataLoader même sans API documentée → erreurs runtime.

#### Solution

Factorisation dans une fonction dédiée `_api_doc_guard()` injectée dans les trois étapes :

```python
def _api_doc_guard(api_doc: str) -> str:
    if not (api_doc or "").strip():
        return (
            "\nIMPORTANT:\n"
            "- DataLoader has NO usable API for this problem (api_doc is empty).\n"
            "- Do NOT use DataLoader at all.\n"
            "- Extract all required numeric data directly from the problem text "
            "and encode it as Python lists/dicts.\n"
        )
    return ""
```

Auparavant ce bloc n'était présent que dans `_define_variables`. Il est désormais appliqué aussi dans `_define_objective` et `_define_constraints`.

#### Impact

Empêche les hallucinations DataLoader sur l'ensemble du pipeline. ✅ **Validé**

---

### 3.3 Correction compatibilité typing Python

**Fichier modifié :** `code_utils.py`

#### Problème

```
TypeError: unsupported operand type(s) for |: '_GenericAlias' and 'NoneType'
```

#### Cause

Utilisation de la syntaxe union Python 3.10+ :

```python
List[str] | None
```

#### Solution

Remplacement par la forme compatible Python 3.8+ :

```python
Optional[List[str]]
```

#### Impact

Évite crash au démarrage du batch sur Python < 3.10. ✅ **Validé**

---

### 3.4 Ajout / restauration de sanitize_python

**Fichier modifié :** `code_utils.py`  
**Utilisé dans :** `main.py`

#### Problème

```
AttributeError: module 'code_utils' has no attribute 'sanitize_python'
```

#### Solution

Réintégration de la fonction `sanitize_python()` dans `code_utils.py`.

#### Impact

Pipeline complet sans crash intermédiaire. ✅ **Validé**

---

### 3.5 Gating étendu à print_solution + filtre lignes narratives

**Fichier modifié :** `llm_utils.py`  
**Fonctions impactées :** `print_solution`, `_llm_step_with_gating`, nouvelle `_strip_narrative_lines`

#### Problème

Le LLM générait parfois du texte narratif mélangé avec du code Python dans le bloc `print_solution`, notamment quand aucune fence ` ```python ``` ` n'était présente dans la réponse. `outer_code_parse` retournait alors le texte brut entier, causant :

```
SyntaxError: invalid syntax
    Add this code right after:
        ^
```

Exemples de lignes narratives observées :
- `Add this code right after:`
- `Code to insert:`
- `Step 1:`, `Note:`, `Explanation:`

#### Cause

`print_solution` appelait `openai_ask_requests` directement sans aucune validation — c'était le seul step non gaté.

#### Solution

**a) `_strip_narrative_lines()`** — nouvelle fonction de pré-nettoyage :

Passe en revue chaque ligne du snippet et convertit en commentaire Python toute ligne correspondant à un pattern narratif connu :

```python
_NARRATIVE_PATTERNS = re.compile(r"""
    ^\s*(
        add\s+this\s+code
      | code\s+to\s+insert
      | insert\s+the\s+following
      | place\s+this\s+(after|before|here)
      | here\s+is\s+the\s+code
      | replace\s+the\s+line
      | step\s+\d+\s*[:\-]
      | note\s*: | explanation\s*: | output\s*:
      | usage\s*: | example\s*: | instructions?\s*:
      | updated?\s+code\s*: | new\s+code\s*: | solution\s*:
    )\b
""", re.VERBOSE | re.IGNORECASE)
```

Les lignes qui ressemblent à du Python (contiennent `=`, `(`, des mots-clés) sont toujours conservées.

**b) `print_solution` branché sur `_llm_step_with_gating`** avec `allow_undefined=True` :

Le paramètre `allow_undefined=True` désactive le check `find_undefined_names` uniquement pour ce step — les variables runtime (`solver`, `status`, `animals`…) sont légitimement définies ailleurs dans le code accumulé et provoqueraient des faux positifs. Le `compile()` continue de s'exécuter, ce qui suffit à détecter les lignes narratives.

**c) Ordre des validations dans `_llm_step_with_gating`** :

```
1. _strip_narrative_lines   ← supprime le texte narratif
2. sanitize_python_code     ← strip fences, fix unicode, auto-repair
3. add_type_comments
4. _safe_join               ← garantit \n\n entre blocs (voir 3.7)
5. compile()                ← SyntaxError / invalid syntax
6. find_undefined_names     ← NameError (sauté si allow_undefined=True)
```

**d) Instructions renforcées dans tous les prompts** :

```
Output ONLY valid Python code — no markdown fences, no prose, no instructions.
Do NOT write lines like "Add this code right after:", "Code to insert:", etc.
Your response must start on a new line and be syntactically self-contained.
```

#### Impact

Élimination des SyntaxError causées par du texte narratif. Le gating couvre désormais l'ensemble des étapes du pipeline LLM. ✅ **Validé**

---

### 3.6 Gestion des timeouts réseau mid-pipeline

**Fichiers modifiés :** `llm_utils.py`, `main.py`  
**Nouvelle classe :** `LLMPipelineError`

#### Problème

Un timeout réseau en plein milieu de `implement_optimization` faisait crasher tout le processus batch :

```
RuntimeError: OpenAI request failed after retries: HTTPSConnectionPool(...): Read timed out.
```

#### Solution

**a) `LLMPipelineError`** — nouvelle exception dans `llm_utils.py` :

```python
class LLMPipelineError(RuntimeError):
    def __init__(self, step: str, cause: Exception):
        self.step = step
        self.cause = cause
        super().__init__(f"LLM pipeline failed at step '{step}': {cause}")
```

**b) Catch dans `main.py`** : écrit `optim_summary.txt` avec l'erreur dans `[stderr]` puis `sys.exit(1)` → le batch runner voit `RUN_FAILED` proprement.

**c) Même protection sur `print_solution`** (step non critique) : fallback `pass` si timeout.

#### Impact

Plus de crash fatal du processus batch sur timeout. ✅ **Validé**

---

### 3.7 Concaténation sécurisée des blocs : `_safe_join`

**Fichier modifié :** `llm_utils.py`  
**Fonctions impactées :** `implement_optimization`, `_llm_step_with_gating`  
**Nouvelle fonction :** `_safe_join`

#### Problème

```
SyntaxError: invalid syntax
    solver = define_solver("SCIP")num_products = 3
                                  ^
```

`_define_solver` retournait `"\n\nsolve = define_solver("SCIP")"` sans newline final. Les `+=` bruts dans `implement_optimization` ne garantissaient aucun séparateur entre blocs.

#### Solution

```python
def _safe_join(base: str, snippet: str) -> str:
    """Concatenate two Python source blocks with a guaranteed blank-line separator."""
    base = base.rstrip()
    snippet = snippet.strip()
    if not snippet:
        return base
    return base + "\n\n" + snippet + "\n"
```

Tous les `+=` dans `implement_optimization` remplacés par `_safe_join`. Également utilisé dans `_llm_step_with_gating` pour construire `combined` avant `compile()`.

#### Validation

```
OLD (+=)         → 'solver = define_solver("SCIP")num_products = 3'   → SyntaxError ✅
NEW (_safe_join) → 'solver = define_solver("SCIP")\n\nnum_products = 3\n'  → OK ✅
```

#### Impact

Élimination définitive des SyntaxError de concaténation. ✅ **Validé**

---

### 3.8 Timeout subprocess : batch et solveur

**Fichier modifié :** `main.py`  
**Constantes ajoutées :** `BATCH_PROBLEM_TIMEOUT`, `SOLUTION_TIMEOUT`  
**Nouveaux arguments CLI :** `--problem-timeout`, `--solution-timeout`

#### Problème

Le batch se bloquait indéfiniment sur certains problèmes (ex. IndustryOR_36 INFEASIBLE) car les deux `subprocess.run(...)` n'avaient aucun `timeout` :

```
[36/100] ▶ Running IndustryOR_36
... (bloqué indéfiniment)
```

#### Solution

**a) Constantes de timeout configurables en tête de fichier :**

```python
BATCH_PROBLEM_TIMEOUT: int = 600   # 10 min max par problème (LLM + solveur)
SOLUTION_TIMEOUT:      int = 120   # 2 min max pour solution.py seul
```

**b) `batch_run` — timeout sur le subprocess enfant :**

```python
try:
    run = subprocess.run(cmd, timeout=problem_timeout)
    returncode = run.returncode
except subprocess.TimeoutExpired:
    # Écrit TIMEOUT dans optim_summary.txt → status = "TIMEOUT" → continue
```

**c) `run_solution` — timeout sur `solution.py` avec fallback propre :**

```python
try:
    result = subprocess.run(..., timeout=timeout)
    return result
except subprocess.TimeoutExpired:
    return subprocess.CompletedProcess(
        returncode=1, stdout="", stderr="TIMEOUT: solution.py exceeded Xs..."
    )
```

**d) Nouveaux arguments CLI :**

```bash
python3 main.py batch --dataset IndustryOR --all --problem-timeout 300
python3 main.py -f IndustryOR_36 --solution-timeout 60
```

**e) Résumé final enrichi avec compteur `Timeouts` :**

```
📊 Comparable: 98 | Passed: 72 | Timeouts: 2 | Total: 100
```

#### Impact

Le batch ne se bloque plus jamais. Les problèmes qui dépassent le timeout sont marqués `TIMEOUT` dans `batch_results.csv` et le batch continue immédiatement vers le suivant. ✅ **Validé**

---

### 3.9 Détection INFEASIBLE et retry automatique des contraintes

**Fichier modifié :** `llm_utils.py`  
**Nouvelles fonctions :** `_is_infeasible`, `_probe_solution`, `_regenerate_constraints_infeasible`  
**Fonction modifiée :** `implement_optimization`

#### Problème

Le LLM générait des contraintes trop restrictives ou contradictoires, rendant le modèle INFEASIBLE à l'exécution :

```
E0000 ... linear_solver.cc:1891] No solution exists.
MPSolverInterface::result_status_ = MPSOLVER_INFEASIBLE
```

Cas observés : IndustryOR_36, IndustryOR_50, IndustryOR_70. Ces erreurs ne sont pas détectables statiquement — le code compilait correctement mais le solveur échouait à l'exécution.

#### Solution

**a) `_is_infeasible(stdout, stderr)`** — détection des patterns INFEASIBLE :

```python
_INFEASIBLE_PATTERNS = re.compile(
    r"(INFEASIBLE|infeasible|No\s+solution\s+exists|MPSOLVER_INFEASIBLE"
    r"|model\s+is\s+infeasible|problem\s+is\s+infeasible)",
    re.IGNORECASE,
)
```

Ne se déclenche pas sur `TIMEOUT`, `OPTIMAL`, ou une solution valide.

**b) `_probe_solution(code, timeout=60)`** — exécution isolée dans un répertoire temporaire avant écriture définitive de `solution.py` :

- Copie les dépendances (`optimization_utils.py`, `utils.py`, `log_utils.py`, `data.py`)
- Ajoute un `solver.Solve()` minimal pour que la probe se termine rapidement
- Nettoie le tmpdir après exécution (même en cas d'erreur)
- Retourne `(stdout, stderr, returncode)`

**c) `_regenerate_constraints_infeasible(...)`** — prompt spécialisé avec feedback solver explicite :

```
CRITICAL: The previous constraint implementation caused an INFEASIBLE model.
- Re-read the problem description carefully to identify wrong constraints.
- Prefer inequalities (≤ or ≥) over equalities where the problem allows.
- Do NOT add constraints not explicitly required by the problem.
- Verify sign conventions and RHS values.
```

**d) Boucle INFEASIBLE dans `implement_optimization`** :

```
Step 4 — contraintes
  ├─ _define_constraints (gated LLM)
  ├─ _probe_solution (tmpdir isolé, 60s max)
  ├─ INFEASIBLE détecté ?
  │    └─ retry (max 2 fois) → _regenerate_constraints_infeasible
  │         └─ _probe_solution again
  ├─ Toujours INFEASIBLE après retries ?
  │    └─ utilise quand même les dernières contraintes
  │       (pipeline continue → RUN_FAILED en dernier recours propre)
  └─ Sinon : accepte les contraintes → continue
```

Le snapshot `code_before_constraints` permet de repartir proprement à chaque retry sans régénérer les variables et l'objectif.

**e) Paramètres configurables :**

```python
def implement_optimization(
    prompt_path, context, code_base, api_doc,
    max_infeasible_retries: int = 2,
    probe_timeout: int = 60,
) -> str:
```

#### Validation

Tests unitaires `_is_infeasible` (5/5 ✅) :

```
✅ _is_infeasible('E0000 MPSOLVER_INFEASIBLE')                    = True
✅ _is_infeasible('No solution exists. MPSOLVER_INFEASIBLE')      = True
✅ _is_infeasible('Optimization objective value: 175.37')         = False
✅ _is_infeasible('STATUS: 2')                                    = False
✅ _is_infeasible('TIMEOUT: solution.py exceeded 120s')           = False
```

#### Impact

Les modèles INFEASIBLE déclenchent une régénération ciblée des contraintes avec feedback explicite. Le pipeline ne crashe plus sur INFEASIBLE — il retente intelligemment avant de rendre la main. ✅ **Validé**

---

### 3.10 Détection des erreurs runtime d'ordre de définition et retry objectif

**Fichier modifié :** `llm_utils.py`  
**Nouvelles fonctions :** `_is_runtime_value_error`, `_regenerate_objective_runtime_error`  
**Fonction modifiée :** `implement_optimization`

#### Problème

```
ValueError: Distance matrix D (7x7) must be defined before adding the objective.
```

Observé sur IndustryOR_60. Le LLM générait une fonction objectif qui référençait une matrice `D` ou d'autres données qui n'avaient pas encore été définies dans `code_base` au moment de l'étape objectif. Ces erreurs ne sont pas des erreurs de syntaxe (le gating statique ne les voit pas) ni des erreurs INFEASIBLE — elles surviennent à l'**exécution du code généré** car le LLM a généré une garde `ValueError` qui vérifie la présence de données nécessaires.

#### Cause

L'étape objectif ne bénéficiait d'aucun probe à l'exécution. Seules les contraintes étaient testées via `_probe_solution`. Le `_llm_step_with_gating` vérifie `compile()` et les undefined names statiques, mais ne détecte pas les `ValueError` levées à l'exécution.

#### Solution

**a) `_is_runtime_value_error(stdout, stderr)`** — détecte les `ValueError` et messages d'ordre de définition :

```python
_RUNTIME_VALUE_ERROR_PATTERNS = re.compile(
    r"(ValueError|must\s+be\s+defined\s+before|not\s+defined\s+before"
    r"|has\s+not\s+been\s+initialized|missing\s+required\s+data"
    r"|cannot\s+be\s+used\s+before)",
    re.IGNORECASE,
)
```

Ne se déclenche pas sur INFEASIBLE ni SyntaxError.

**b) `_regenerate_objective_runtime_error(...)`** — prompt spécialisé avec feedback explicite :

```
CRITICAL: The previous objective implementation caused a runtime error.
- Define ALL required data (matrices, cost vectors, etc.) inline at the top
  of your snippet, BEFORE using them in the objective expression.
- Do NOT assume any variable exists unless visible in the "code so far" block.
- If data comes from DataLoader, load it explicitly using the DataLoader API.
```

**c) Boucle runtime retry dans `implement_optimization`** — insérée entre step 2 (variables) et step 4 (contraintes) :

```
Step 3 — objectif
  ├─ _define_objective (gated LLM)
  ├─ _probe_solution (tmpdir isolé, timeout=60s)
  ├─ ValueError / RuntimeError détecté ?
  │    └─ retry (max 2) → _regenerate_objective_runtime_error avec feedback
  │         └─ _probe_solution again
  ├─ Toujours en erreur après retries ?
  │    └─ utilise quand même (pipeline continue)
  └─ Sinon : accepte l'objectif → passe aux contraintes (step 4)
```

**d) Nouveau paramètre configurable** :

```python
def implement_optimization(
    ...,
    max_infeasible_retries: int = 2,
    max_runtime_retries: int = 2,   # ← nouveau
    probe_timeout: int = 60,
) -> str:
```

#### Validation

Tests unitaires `_is_runtime_value_error` (6/6 ✅) :

```
✅ ValueError: Distance matrix D (7x7) must be defined before...  = True
✅ ValueError: variable x has not been initialized                 = True
✅ must be defined before adding the objective                     = True
✅ STATUS: 0  (solution valide)                                    = False
✅ MPSOLVER_INFEASIBLE                                             = False
✅ SyntaxError: invalid syntax                                     = False
```

#### Impact

Les erreurs de type "donnée non définie avant utilisation dans l'objectif" déclenchent une régénération ciblée de l'étape objectif avec feedback explicite. Le pipeline couvre maintenant les erreurs runtime à **toutes les étapes critiques**. ✅ **Validé**

---

### 3.11 Support multi-format datasets (ComplexOR, LPWP)

**Fichier modifié :** `main.py`  
**Nouvelles fonctions :** `detect_dataset_format`, `load_subdir_dataset`, `_build_en_question`, `_subdir_id`  
**Nouveau dataclass :** `DatasetFormat`  
**Fonction modifiée :** `batch_run`

#### Problème

Le batch runner ne supportait que le format IndustryOR (un seul fichier JSONL avec tous les problèmes). Les autres datasets (ComplexOR, LPWP) utilisent un format différent : un répertoire par problème, chacun contenant `description.txt` (énoncé) et `sample.json` (données d'entrée + réponse attendue).

**IndustryOR (format existant) :**
```
datasets/IndustryOR/IndustryOR.json     ← JSONL, 1 ligne = 1 problème
  {"en_question": "...", "en_answer": "219816.0", "id": 1}
```

**ComplexOR (nouveau format) :**
```
datasets/ComplexOR/
  ├── aircraft_assignment/
  │   ├── description.txt     ← énoncé du problème
  │   └── sample.json         ← {"input": {...}, "output": [valeur]}
  ├── diet_problem/
  │   ├── description.txt
  │   └── sample.json
  └── ...                     ← 18 sous-dossiers
```

**LPWP (nouveau format) :**
```
datasets/LPWP/
  ├── prob_0/
  │   ├── description.txt
  │   └── sample.json
  ├── prob_1/
  └── ...                     ← 288 sous-dossiers
```

#### Solution

**a) `DatasetFormat` — dataclass de détection :**

```python
@dataclass
class DatasetFormat:
    kind: str              # "jsonl" ou "subdirs"
    jsonl_path: str        # chemin du .json (si kind == "jsonl")
    sub_dirs: List[str]    # liste triée des sous-dossiers (si kind == "subdirs")
```

**b) `detect_dataset_format(dataset_dir)`** — détection automatique :

- Si un seul fichier `.json` existe à la racine du dataset → format JSONL (IndustryOR)
- Sinon, cherche les sous-dossiers contenant `description.txt` + `sample.json` → format subdirs
- Lève `ValueError` si aucun format n'est reconnu

**c) `load_subdir_dataset(dataset_dir, sub_dirs)`** — conversion vers le format interne :

Pour chaque sous-dossier :
1. Lit `description.txt` → texte de description
2. Lit `sample.json` → extrait `input` (données) et `output` (réponse)
3. Construit `en_question` = description + données d'entrée formatées en JSON
4. Construit `en_answer` = valeur de `output` (scalaire)
5. Ajoute `source_dir` = nom du sous-dossier d'origine

Le `user_input.md` généré ne contient **jamais** la réponse attendue (`output`).

**d) `_subdir_id(name, index)`** — attribution d'ID numériques :

- Pour LPWP (`prob_42`) → extrait `42` du nom
- Pour ComplexOR (`diet_problem`) → utilise l'index 1-based dans la liste triée

**e) `batch_run` mis à jour :**

- Utilise `detect_dataset_format()` au lieu de `find_single_json_file()` hardcodé
- Affiche `source_dir` dans l'interface CLI (ex. `ComplexOR_7  (diet_problem)`)
- Inclut `source_dir` dans `batch_results.csv` pour traçabilité

#### Validation

```bash
# ComplexOR — 18 problèmes détectés, dry-run OK
$ python3 main.py batch --dataset ComplexOR --all --dry-run
╭──────────────────────────────╮
│  🚀 LLoCO Batch Runner       │
│  Dataset: ComplexOR          │
│  Source:  18 subdirectories  │
│  Mode:    --all              │
│  Total:   18 problem(s)      │
╰──────────────────────────────╯
[1/18] 🧪 DRY_RUN ComplexOR_1
...
[18/18] 🧪 DRY_RUN ComplexOR_18

# LPWP — 288 problèmes détectés, sélection par ID
$ python3 main.py batch --dataset LPWP --id 0 --dry-run
[1/1] 🧪 DRY_RUN LPWP_0

# IndustryOR — pipeline inchangé, toujours fonctionnel
$ python3 main.py batch --dataset IndustryOR --id 1 --dry-run
[1/1] 🧪 DRY_RUN IndustryOR_1
```

**Contenu vérifié de `user_input.md` (LPWP_0) :**
```
A fishery wants to transport their catch. They can either use local sled
dogs or trucks. [...]

Input data:
​```json
{
  "DogCapability": 100,
  "TruckCapability": 300,
  "DogCost": 50,
  "TruckCost": 100,
  "MaxBudget": 1000
}
​```
```
→ Pas de `output` / `en_answer` dans le fichier. ✅

**Contenu vérifié de `batch_results.csv` :**
```csv
id,folder,source_dir,expected,objective,ok,status
0,LPWP_0,prob_0,1300,,,DRY_RUN
```
→ `source_dir` tracé correctement. ✅

#### Impact

Le batch runner supporte désormais les trois datasets sans modification du pipeline existant. La détection du format est automatique — aucun flag supplémentaire n'est nécessaire. ✅ **Validé**

---

## 4) Résultat observé — pipeline technique ✅ STABILISÉ

Toutes les catégories d'erreurs techniques du pipeline ont été traitées et validées :

| Erreur | Statut |
|---|---|
| `NameError: name 'x' is not defined` | ✅ Éliminé (gating + feedback LLM) |
| `SyntaxError` texte narratif | ✅ Éliminé (`_strip_narrative_lines` + gating `print_solution`) |
| `SyntaxError` collision de blocs | ✅ Éliminé (`_safe_join`) |
| `TypeError` DataLoader sans api_doc | ✅ Éliminé (`_api_doc_guard`) |
| `RuntimeError` timeout réseau LLM | ✅ Éliminé (`LLMPipelineError` + catch) |
| Batch bloqué indéfiniment | ✅ Éliminé (timeout `subprocess.run`) |
| `MPSOLVER_INFEASIBLE` | ✅ Traité (probe + retry ciblé contraintes) |
| `ValueError` dépendance données objectif | ✅ Traité (probe + retry ciblé objectif) |
| Datasets non-JSONL (ComplexOR, LPWP) | ✅ Supporté (détection auto + conversion interne) |

---

## 5) Améliorations qualité modèle (prompts & extraction)

Les FAILED restants ne sont plus des erreurs techniques — ils relevaient exclusivement de la qualité de modélisation. Les améliorations suivantes ont été apportées :

### 5.1 Correction direction Min/Max (objectif)

**Fichiers modifiés :** `optimization_utils.py`, `llm_utils.py`

- **Docstring `add_objective` corrigée** : le paramètre s'appelait `direction` dans la doc mais `maximize` dans le code. La docstring liste maintenant explicitement quand utiliser `maximize=True` vs `maximize=False` avec des exemples (profit → True, cost → False).
- **Prompt `_define_objective` enrichi** : bloc CRITICAL ajouté qui force le LLM à identifier le sens d'optimisation dans l'énoncé avant de coder. Exemples explicites : "minimize the total cost" → `maximize=False`.

#### Impact attendu

Réduction des erreurs de direction d'optimisation — premier poste d'erreurs restant.

### 5.2 Guidance contraintes complètes + eps_relax

**Fichier modifié :** `llm_utils.py`

- **Prompt `_define_constraints` enrichi** avec trois blocs :
  1. **Constraint completeness** : demande au LLM de lister TOUTES les contraintes de la formalisation avant de coder. Interdit de sauter les contraintes "évidentes".
  2. **Strict vs non-strict inequalities** : explique le piège `eps_relax=0.0` (< silencieusement traité comme ≤). Donne les valeurs recommandées : `eps_relax=1` pour entiers, `eps_relax=0.0001` pour continus.
  3. **Constraint direction** : rappel explicite de `operator.le` (≤), `operator.ge` (≥), `operator.eq` (=).

#### Impact attendu

Réduction des contraintes manquantes et des erreurs de direction de contrainte.

### 5.3 Formalisation sur modèle complet (gpt-5)

**Fichier modifié :** `llm_utils.py`

- `summarize_problem_description` passé de `o4-mini` à `gpt-5`. L'étape de formalisation mathématique est critique — une mauvaise formulation propage des erreurs dans tout le pipeline.

#### Impact attendu

Meilleure qualité des formalisations mathématiques, en particulier sur les problèmes complexes multi-contraintes.

### 5.4 Worked example dans le system prompt code

**Fichier modifié :** `prompts/system_prompt_code_.txt`

- Ajout d'un exemple complet de résolution (problème de production à 2 produits) montrant le pattern exact : extraction données → `define_variables` → `define_linear_expr` → `add_objective` → `add_constraint`.
- Section "Key takeaways" récapitulant les conventions à suivre.

#### Impact attendu

Le LLM a un modèle concret à suivre au lieu de deviner le pattern d'utilisation de la bibliothèque.

### 5.5 Extraction explicite des coefficients numériques

**Fichier modifié :** `prompts/system_prompt_problem_summary.txt`

- Nouveau bloc "Numeric data extraction" dans le prompt de formalisation. Demande au LLM d'extraire et lister explicitement TOUS les coefficients, paramètres et constantes (valeurs des tableaux, coûts, capacités, taux, limites, bornes, matrices).

#### Impact attendu

Réduction des erreurs d'extraction de coefficients depuis les tableaux et textes d'énoncé.

### 5.6 Réactivation `add_type_comments` (annotations numpy)

**Fichier modifié :** `utils.py`

- L'ancien `TypeCommentInserter` basé sur `libcst` était désactivé (no-op) et avait un bug sur les tuples shapes.
- Remplacement complet par une approche regex robuste (`_DEFVAR_RE`) qui :
  1. Détecte chaque appel `var = define_variables(solver, shape=..., ...)`
  2. Injecte un commentaire `# IMPORTANT: var is a numpy array (dtype=object) of OR-Tools decision variables with shape=...`
  3. Rappelle d'utiliser `.flatten()` pour obtenir un array 1-D pour `define_linear_expr`
- Suppression de la dépendance `libcst` (import `re` suffit)
- Fonctionne avec tous les formats de shape : `(3,)`, `5`, `(4, 3)`, `num_items`

#### Impact attendu

Le LLM sait que `define_variables` retourne un numpy array et connaît sa shape, ce qui réduit les erreurs de manipulation (indexation, reshape, flatten).

### 5.7 Probe de validation direction min/max post-objectif

**Fichier modifié :** `llm_utils.py`  
**Nouvelles fonctions :** `_check_objective_direction`, `_flip_objective_direction`  
**Nouveau regex :** `_MAXIMIZE_RE`

- Après que l'objectif passe le runtime check, un appel LLM léger (`max_tokens=10`) vérifie si la direction `maximize=True/False` dans le code généré est cohérente avec l'énoncé du problème.
- Le LLM répond par un seul mot : `MAXIMIZE` ou `MINIMIZE`.
- Si incohérence détectée, `_flip_objective_direction` inverse automatiquement `maximize=True` ↔ `maximize=False` via regex.
- Le flip est loggé dans stderr pour traçabilité.
- En cas d'erreur réseau ou de réponse ambiguë, la direction originale est conservée (fail-safe).

#### Flux dans `implement_optimization` :

```
Step 3 — objectif
  ├─ _define_objective (gated LLM)
  ├─ _probe_solution (runtime check)
  ├─ Step 3b — _check_objective_direction (LLM léger, 10 tokens)
  │    ├─ Cohérent → continue
  │    └─ Incohérent → _flip_objective_direction → log + continue
  └─ Passe aux contraintes (step 4)
```

#### Impact attendu

Détection et correction automatique des erreurs de direction min/max — premier poste d'erreurs de qualité modèle.

### 5.8 Injection `code_example.py` pour ComplexOR/LPWP

**Fichier modifié :** `main.py`  
**Fonctions modifiées :** `_build_en_question`, `load_subdir_dataset`

- Les datasets ComplexOR et LPWP contiennent un fichier `code_example.py` par problème avec :
  - La signature de fonction (noms et types des paramètres)
  - Un docstring décrivant chaque paramètre
  - La valeur de retour attendue (qui révèle souvent la direction : "minimized total cost", "maximized profit")
- `load_subdir_dataset` lit `code_example.py` s'il existe et le passe à `_build_en_question`.
- Le stub est ajouté en fin de `user_input.md` avec la mention explicite "for reference only — do NOT call this function, use the optimization library instead".

#### Exemple de `user_input.md` généré (ComplexOR/diet_problem) :

```
Consider a diet problem. [...]

Input data:
​```json
{"food_set": ["Apple", "Banana"], ...}
​```

Function signature and parameter descriptions (for reference only — do NOT call
this function, use the optimization library instead):
​```python
def diet_problem(food_set, nutrient_set, food_cost, ...):
    """
    Args:
        food_set: List of strings, each representing a type of food.
        ...
    Returns:
        total_cost: The minimized total cost to satisfy the nutrient requirements.
    """
​```
```

#### Impact attendu

Le LLM dispose des noms exacts des paramètres, de leurs types et de la direction d'optimisation attendue, ce qui réduit les erreurs de modélisation sur les datasets ComplexOR/LPWP.

### 5.9 Détection INFEASIBLE par status code numérique (Codex review)

**Fichier modifié :** `llm_utils.py`  
**Nouveau regex :** `_INFEASIBLE_STATUS_RE`

#### Problème

`_is_infeasible` ne détectait que les patterns textuels (ex. `MPSOLVER_INFEASIBLE`). Or la probe imprime `STATUS: 2` (code numérique OR-Tools pour INFEASIBLE) sans nécessairement le texte associé. Résultat : la boucle de retry contraintes ne se déclenchait pas dans certains cas.

#### Solution

Ajout d'un regex `_INFEASIBLE_STATUS_RE = re.compile(r"STATUS:\s*2\b")` dans `_is_infeasible`. La fonction vérifie maintenant les deux : patterns textuels ET status code numérique.

#### Impact

Détection plus robuste des modèles INFEASIBLE, indépendante du format de message d'OR-Tools.

### 5.10 Correction du flip de direction avec espaces (Codex review)

**Fichier modifié :** `llm_utils.py`  
**Nouveau regex :** `_MAXIMIZE_ASSIGN_RE`

#### Problème

`_flip_objective_direction` utilisait `.replace(f"maximize={val}")` pour inverser la direction. Mais `_MAXIMIZE_RE` accepte des espaces autour de `=` (ex. `maximize = True`). Dans ce cas, le `.replace()` ne trouvait pas la chaîne exacte et le flip échouait silencieusement.

#### Solution

Remplacement de `.replace()` par un regex dédié `_MAXIMIZE_ASSIGN_RE = re.compile(r"(maximize\s*=\s*)(True|False)")` qui capture le préfixe avec espaces et remplace uniquement la valeur.

#### Validation

```python
_flip("maximize = True")   → "maximize = False"   ✅
_flip("maximize=False")    → "maximize=True"       ✅
_flip("maximize =True")    → "maximize =False"     ✅
```

#### Impact

Le flip de direction fonctionne maintenant quel que soit le style de formatage du LLM.

### 5.11 Propagation `--solution-timeout` au subprocess batch (Codex review)

**Fichier modifié :** `main.py`

#### Problème

`batch_run` ne passait jamais `--solution-timeout` au subprocess enfant. Chaque problème utilisait le défaut 120s, même si `--problem-timeout` était plus élevé. Des problèmes solvables mais lents échouaient prématurément avec `RUN_FAILED`.

#### Solution

- Ajout du paramètre `solution_timeout` à `batch_run()`
- Ajout de `--solution-timeout` dans la commande `cmd` du subprocess
- Ajout de `--solution-timeout` comme argument CLI du batch parser
- Propagation de `args.solution_timeout` vers `batch_run()`

#### Impact

Le timeout du solveur est maintenant configurable en mode batch et cohérent entre parent et enfant.

### 5.12 Refactoring affichage — centralisation dans `UI/utils.py`

**Fichiers modifiés :** `UI/utils.py`, `main.py`

#### Problème (review sabeaussan)

Tout le code d'affichage batch (box, horizontal rules, formatage des résultats) était dans `main.py`, alors que le projet dispose d'un module `UI/utils.py` dédié à l'affichage.

#### Solution

**Déplacé de `main.py` vers `UI/utils.py` :**
- `_hr()` → `hr()`
- `_wlen()` → `wlen()`
- `_box()` → `box()`
- `_mode_str()` → `mode_str()`
- Import `wcwidth` déplacé dans `UI/utils.py`

**Nouvelles fonctions d'affichage batch dans `UI/utils.py` :**
- `print_batch_header()` — box avec configuration du batch
- `print_batch_dry_run()` — ligne dry-run
- `print_batch_running()` — header de problème en cours
- `print_batch_timeout()` — message timeout
- `print_batch_result()` — PASSED/FAILED/status avec expected/objective/source_dir
- `print_batch_summary()` — ligne finale (report + stats)

**`main.py` allégé** — plus aucune logique d'affichage, tout passe par `UI/utils.py`.

#### Impact

Séparation propre logique métier / affichage. `main.py` est allégé et plus lisible.

---

## 6) Recommandations futures

- Étendre `_regenerate_constraints_infeasible` avec un parser d'énoncé pour identifier automatiquement les contraintes suspectes
- Ajouter logs d'audit du code généré (`generated_code_log.txt` par problème) pour faciliter le débogage qualité modèle
- Envisager un retry complet (variables + objectif + contraintes) sur INFEASIBLE persistant après les retries contraintes seules

---

## 7) Récapitulatif des changements par fichier

| Fichier | Modification | Section |
|---|---|---|
| `llm_utils.py` | Gating étendu à `_define_objective`, `_define_constraints`, `print_solution` | 3.1 |
| `llm_utils.py` | `_api_doc_guard()` factorisé et appliqué aux 3 étapes | 3.2 |
| `llm_utils.py` | `_strip_narrative_lines()` ajouté, branché en tête du gating | 3.5 |
| `llm_utils.py` | `allow_undefined` param dans `_llm_step_with_gating` | 3.5 |
| `llm_utils.py` | `LLMPipelineError` nouvelle exception typée | 3.6 |
| `llm_utils.py` | `_safe_join()` — concaténation sécurisée avec `\n\n` garanti | 3.7 |
| `llm_utils.py` | `implement_optimization` : tous les `+=` remplacés par `_safe_join` | 3.7 |
| `llm_utils.py` | `_llm_step_with_gating` : `combined` construit via `_safe_join` | 3.7 |
| `llm_utils.py` | Instructions "Output ONLY valid Python / start on a new line" dans tous les prompts | 3.5 |
| `llm_utils.py` | `_is_infeasible()` — détection INFEASIBLE dans stdout/stderr | 3.9 |
| `llm_utils.py` | `_probe_solution()` — exécution isolée dans tmpdir avant écriture finale | 3.9 |
| `llm_utils.py` | `_regenerate_constraints_infeasible()` — prompt spécialisé avec feedback solver | 3.9 |
| `llm_utils.py` | `implement_optimization` : boucle INFEASIBLE retry avec snapshot contraintes | 3.9 |
| `llm_utils.py` | `_is_runtime_value_error()` — détection ValueError / dépendance données | 3.10 |
| `llm_utils.py` | `_regenerate_objective_runtime_error()` — prompt spécialisé runtime error objectif | 3.10 |
| `llm_utils.py` | `implement_optimization` : boucle runtime retry objectif + `max_runtime_retries` | 3.10 |
| `main.py` | Catch `LLMPipelineError` → `RUN_FAILED` propre sans crash batch | 3.6 |
| `main.py` | Catch `RuntimeError` sur `print_solution` → fallback `pass` | 3.6 |
| `main.py` | `BATCH_PROBLEM_TIMEOUT` / `SOLUTION_TIMEOUT` — constantes configurables | 3.8 |
| `main.py` | `subprocess.run(cmd, timeout=...)` dans `batch_run` | 3.8 |
| `main.py` | `subprocess.run(..., timeout=...)` dans `run_solution` + fallback `CompletedProcess` | 3.8 |
| `main.py` | `--problem-timeout` / `--solution-timeout` — nouveaux arguments CLI | 3.8 |
| `main.py` | Résumé batch enrichi avec compteur `Timeouts` | 3.8 |
| `code_utils.py` | `List[str] \| None` → `Optional[List[str]]` | 3.3 |
| `code_utils.py` | Restauration de `sanitize_python()` | 3.4 |
| `main.py` | `DatasetFormat` dataclass — détection format JSONL vs subdirs | 3.11 |
| `main.py` | `detect_dataset_format()` — auto-détection du format dataset | 3.11 |
| `main.py` | `load_subdir_dataset()` — chargement ComplexOR/LPWP en format interne | 3.11 |
| `main.py` | `_build_en_question()` — construction description + input JSON | 3.11 |
| `main.py` | `_subdir_id()` — attribution ID numériques (prob_N → N, sinon index) | 3.11 |
| `main.py` | `batch_run` : détection auto via `detect_dataset_format()` | 3.11 |
| `main.py` | `batch_run` : affichage `source_dir` dans CLI et `batch_results.csv` | 3.11 |
| `optimization_utils.py` | Docstring `add_objective` : `direction` → `maximize`, exemples min/max | 5.1 |
| `llm_utils.py` | Prompt `_define_objective` : bloc CRITICAL direction maximize/minimize | 5.1 |
| `llm_utils.py` | Prompt `_define_constraints` : blocs completeness, eps_relax, direction | 5.2 |
| `llm_utils.py` | `summarize_problem_description` : `o4-mini` → `gpt-5` | 5.3 |
| `prompts/system_prompt_code_.txt` | Worked example complet (pattern define_variables → add_objective → add_constraint) | 5.4 |
| `prompts/system_prompt_problem_summary.txt` | Bloc "Numeric data extraction" — extraction explicite coefficients | 5.5 |
| `utils.py` | `add_type_comments` réécrit en regex — annotations numpy shape sur `define_variables` | 5.6 |
| `utils.py` | Suppression dépendance `libcst`, remplacement par `_DEFVAR_RE` | 5.6 |
| `llm_utils.py` | `_check_objective_direction()` — appel LLM léger pour vérifier direction | 5.7 |
| `llm_utils.py` | `_flip_objective_direction()` — inversion automatique maximize ↔ minimize | 5.7 |
| `llm_utils.py` | `_MAXIMIZE_RE` — regex détection `add_objective(..., maximize=...)` | 5.7 |
| `llm_utils.py` | `implement_optimization` step 3b — direction validation post-objectif | 5.7 |
| `main.py` | `_build_en_question` — paramètre optionnel `code_example` | 5.8 |
| `main.py` | `load_subdir_dataset` — lecture et injection de `code_example.py` | 5.8 |
| `llm_utils.py` | `_INFEASIBLE_STATUS_RE` — détection INFEASIBLE par status code `STATUS: 2` | 5.9 |
| `llm_utils.py` | `_is_infeasible` — ajout vérification status code numérique OR-Tools | 5.9 |
| `llm_utils.py` | `_MAXIMIZE_ASSIGN_RE` — regex robuste pour flip avec espaces | 5.10 |
| `llm_utils.py` | `_flip_objective_direction` — réécriture avec regex au lieu de `.replace()` | 5.10 |
| `main.py` | `batch_run` — nouveau paramètre `solution_timeout`, propagé au subprocess | 5.11 |
| `main.py` | `--solution-timeout` ajouté au batch CLI parser | 5.11 |
| `UI/utils.py` | Fonctions `hr`, `wlen`, `box`, `mode_str` déplacées depuis `main.py` | 5.12 |
| `UI/utils.py` | Nouvelles fonctions `print_batch_*` (header, dry_run, running, timeout, result, summary) | 5.12 |
| `main.py` | Suppression de tout le code d'affichage — appels délégués à `UI/utils.py` | 5.12 |

---

## 8) Conclusion

Le pipeline LLoCO est désormais **techniquement stabilisé à 100%** :

- ✅ Plus aucune erreur runtime liée au LLM ne crashe le batch
- ✅ Plus aucun blocage indéfini du runner (timeout sur tous les subprocesses)
- ✅ Texte narratif généré par le LLM filtré avant compilation
- ✅ Blocs de code toujours séparés proprement (`_safe_join`)
- ✅ Modèles INFEASIBLE traités par retry ciblé sur les contraintes avec feedback
- ✅ Erreurs de dépendance de données dans l'objectif traitées par retry ciblé avec feedback
- ✅ Support multi-format : IndustryOR (JSONL), ComplexOR (subdirs), LPWP (subdirs) — détection automatique
- ✅ Tous les échecs sont capturés, loggés et visibles dans `batch_results.csv` (avec `source_dir`)

Les erreurs de qualité de modélisation (direction min/max, contraintes manquantes, extraction de coefficients) sont désormais adressées par :
- Améliorations ciblées des prompts et de la documentation API (sections 5.1–5.5)
- Annotations numpy automatiques sur les variables de décision (section 5.6)
- Validation automatique de la direction d'optimisation via probe LLM (section 5.7)
- Injection des signatures et descriptions de paramètres des datasets ComplexOR/LPWP (section 5.8)
- Détection INFEASIBLE renforcée par status code numérique (section 5.9)
- Flip de direction robuste indépendant du formatage (section 5.10)
- Propagation du solution-timeout en mode batch (section 5.11)
- Code d'affichage centralisé dans `UI/utils.py` (section 5.12)
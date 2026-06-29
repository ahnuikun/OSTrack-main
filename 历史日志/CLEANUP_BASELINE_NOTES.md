# Cleanup Baseline Notes

- Time: 2026-06-23 14:09:13
- Workspace: D:\PyCharm\Projects\OSTrack-main
- Local baseline source: D:\PyCharm\Projects\OSTrack-4
- Archive: D:\PyCharm\Projects\OSTrack-main\tmp\cleanup_archive_20260623_140913
- Mode: archive old experiment state, restore OSTrack baseline code/configs

## Archived items
- EXPERIMENT_PLAN.md
- FRONT_HISTORY_EXPERIMENT.md
- EXPERIMENT_PLAN_HARD_DISTRACTOR_ARCHIVE.md
- EXPERIMENT_RUNBOOK.md
- MODULE_DESIGN.md
- lib\models\ostrack\candidate_template.py
- lib\models\ostrack\front_history.py
- lib\models\ostrack\temporal_query.py
- tracking\validate_front_history.py
- tracking\validate_temporal_query.py
- experiments\ostrack
- output
- tmp\doc_review
- tmp\pdfs

## Restored baseline files from OSTrack-4
- experiments\ostrack\*
- lib\config\ostrack\config.py
- lib\models\ostrack\base_backbone.py
- lib\models\ostrack\ostrack.py
- lib\models\ostrack\vit_ce.py
- lib\train\base_functions.py
- lib\train\actors\ostrack.py
- lib\train\data\sampler.py
- lib\train\data\processing.py
- lib\test\tracker\ostrack.py

## Fresh directories
- output

## Removed caches
- lib\__pycache__
- lib\config\__pycache__
- lib\config\ostrack\__pycache__
- lib\models\__pycache__
- lib\models\layers\__pycache__
- lib\models\ostrack\__pycache__
- lib\test\__pycache__
- lib\test\analysis\__pycache__
- lib\test\evaluation\__pycache__
- lib\test\parameter\__pycache__
- lib\test\tracker\__pycache__
- lib\test\utils\__pycache__
- lib\train\__pycache__
- lib\train\actors\__pycache__
- lib\train\admin\__pycache__
- lib\train\data\__pycache__
- lib\train\dataset\__pycache__
- lib\train\trainers\__pycache__
- lib\utils\__pycache__
- lib\vis\__pycache__
- tracking\__pycache__

## Path warning
OSTrack-4 was used only as a source-code baseline. Local path files were not blindly copied from it except the files listed above. Dataset/root paths still require verification in OSTrack-main before training.

## Post-clean validation
- Re-copied baseline YAML files from `D:\PyCharm\Projects\OSTrack-4\experiments\ostrack` into `D:\PyCharm\Projects\OSTrack-main\experiments\ostrack` after confirming the first wildcard copy left the directory empty.
- Verified old temporal/front-history/candidate keywords no longer appear under active `lib`, `tracking`, or `experiments` files.
- Verified train and test local paths point to `D:\PyCharm\Projects\OSTrack-main\output`, not `OSTrack-4` or a server path.
- Verified key OSTrack files compile and the baseline build function imports successfully.

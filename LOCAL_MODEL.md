# Use your Colab model

The supplied detection and OBB checkpoints are now integrated through a shared
ensemble. See [models/README.md](models/README.md) for the file mapping,
configuration, and the missing 60-epoch checkpoint details.

The separate needle-detection OBB run and its validation metrics are recorded in
[training_results/needle_obb/README.md](training_results/needle_obb/README.md).
Save its downloaded checkpoint in that folder as `best.pt`.

Download `/content/runs/detect/train/weights/best.pt` from Colab and copy it
beside `main_dual_gauge_reader.py`. A `best.pt` file already exists here;
replace it if your downloaded checkpoint is newer.

Run from the project directory:

```powershell
python -m pip install -r requirements.txt
python main_dual_gauge_reader.py
```

Local YOLO is now the default. No Roboflow API key is required for this mode.
The live reader, photo evaluator, and angle calibrator use the same loader.
Relative model paths are resolved against the project directory.

Optional settings in `.env`:

```dotenv
MODEL_BACKEND=local
LOCAL_MODEL_PATH=best.pt
```

To return to the previous models, set `MODEL_BACKEND=roboflow` and provide
`ROBOFLOW_API_KEY` in `.env`.

The local adapter uses `gauge` and `pointer` boxes. The existing vision
algorithms estimate the pivot and tip and the selected gauge profile converts
the angle to pressure. `maximum`, `minimum`, `pointer_centre`, and `pointer_end`
are ignored: the supplied validation log shows poor detection for those
classes. Detection mAP is not pressure-reading accuracy; measure readings with
labelled gauge photos as described in ACCURACY_TESTING.md.

```powershell
python evaluate_accuracy.py accuracy_data/unijin_labels.csv --profile UNIJIN
```

Fill the `expected` column with real reference readings before evaluation.

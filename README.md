# OCR Assessment Starter

This starter project is a simple structure for the OCR intern assignment.
It is designed around the recommended setup:

- Dataset: IAM handwriting dataset (line-level samples)
- Baseline model: `microsoft/trocr-small-handwritten`
- Training: fine-tune the pretrained model
- Evaluation: CER and WER
- Post-processing: simple prompt-driven or rule-based cleanup

## Project Layout

```text
ocr-assessment-starter/
  README.md
  requirements.txt
  report_template.md
  .gitignore
  src/
    common.py
    prepare_iam.py
    train_trocr.py
    evaluate_trocr.py
    postprocess_llm.py
```

## What Each File Does

- `src/prepare_iam.py`
  Parses IAM metadata, creates a clean manifest CSV, and writes a short data audit.
- `src/train_trocr.py`
  Fine-tunes TrOCR on the train split and evaluates on the validation split.
- `src/evaluate_trocr.py`
  Runs inference on a split and writes metrics plus sample predictions.
- `src/postprocess_llm.py`
  Creates a simple post-processing comparison and a prompt template you can show in the report.
- `report_template.md`
  A clean report outline that matches the company brief.

## Expected Dataset Layout

Place the IAM dataset somewhere like this:

```text
data/raw/iam/
  ascii/
    lines.txt
  lines/
    a01/
      a01-000u/
        a01-000u-00.png
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## If You Only Want To Run OCR

If you do not want to prepare IAM or fine-tune anything, you can run the pretrained model directly on your own images:

```bash
python src/run_pretrained_ocr.py --input-path sample_images --output-csv outputs/pretrained_ocr_predictions.csv --recursive
```

This will download `microsoft/trocr-small-handwritten` the first time and save predictions to `outputs/pretrained_ocr_predictions.csv`.

2. Prepare the manifest and audit:

```bash
python src/prepare_iam.py --iam-root data/raw/iam --manifest data/processed/iam_lines_manifest.csv --audit outputs/data_audit.md --lowercase
```

3. Run a baseline evaluation with the pretrained model:

```bash
python src/evaluate_trocr.py --manifest data/processed/iam_lines_manifest.csv --model-path microsoft/trocr-small-handwritten --split test --output-csv outputs/baseline_predictions.csv --metrics-json outputs/baseline_metrics.json --samples-md outputs/baseline_samples.md --lowercase
```

4. Fine-tune the model:

```bash
python src/train_trocr.py --manifest data/processed/iam_lines_manifest.csv --model-name microsoft/trocr-small-handwritten --output-dir outputs/trocr-small-finetuned --epochs 2 --batch-size 4 --learning-rate 5e-5 --lowercase
```

5. Evaluate the fine-tuned model:

```bash
python src/evaluate_trocr.py --manifest data/processed/iam_lines_manifest.csv --model-path outputs/trocr-small-finetuned --split test --output-csv outputs/finetuned_predictions.csv --metrics-json outputs/finetuned_metrics.json --samples-md outputs/finetuned_samples.md --lowercase
```

6. Create a post-processing comparison:

```bash
python src/postprocess_llm.py --predictions-csv outputs/finetuned_predictions.csv --output-csv outputs/postprocessed_predictions.csv --metrics-json outputs/postprocess_metrics.json --examples-md outputs/postprocess_examples.md
```

## Notes

- For a strong submission, do not skip the baseline.
- If your laptop is weak, run the training script in Colab or Kaggle.
- If you only have CPU, use `--max-train-samples` first to validate the pipeline.
- The post-processing script is intentionally simple so you can explain it clearly in the report.

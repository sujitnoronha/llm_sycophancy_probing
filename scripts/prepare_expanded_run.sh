#!/bin/bash
# Prepare for expanded 200-question benchmark run.
# Backs up old 50-question results and clears checkpoints for baseline + behavioral.
# Probing/steering/MMLU checkpoints are left intact (supplementary analysis on original 50).

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS="$PROJECT_ROOT/outputs"
CHECKPOINTS="$OUTPUTS/checkpoints"

echo "=== Preparing Expanded 200-Question Run ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# 1. Back up old results
BACKUP_DIR="$OUTPUTS/backup_50q_run"
if [ -d "$BACKUP_DIR" ]; then
    echo "Backup already exists at $BACKUP_DIR — skipping backup"
else
    echo "Backing up old 50-question results..."
    mkdir -p "$BACKUP_DIR/baseline"
    mkdir -p "$BACKUP_DIR/behavioral"
    mkdir -p "$BACKUP_DIR/checkpoints"

    # Back up baseline results
    cp "$OUTPUTS"/baseline/*_baseline.json "$BACKUP_DIR/baseline/" 2>/dev/null || true
    echo "  Backed up baseline results"

    # Back up behavioral results
    cp "$OUTPUTS"/behavioral/*_conversations.jsonl "$BACKUP_DIR/behavioral/" 2>/dev/null || true
    cp "$OUTPUTS"/behavioral/*_scores.csv "$BACKUP_DIR/behavioral/" 2>/dev/null || true
    echo "  Backed up behavioral results"

    # Back up relevant checkpoints
    cp "$CHECKPOINTS"/baseline_*.done "$BACKUP_DIR/checkpoints/" 2>/dev/null || true
    cp "$CHECKPOINTS"/behavioral_*.done "$BACKUP_DIR/checkpoints/" 2>/dev/null || true
    echo "  Backed up checkpoints"
fi

echo ""

# 2. Clear baseline checkpoints (need to re-validate all 200 questions)
echo "Clearing baseline checkpoints..."
rm -f "$CHECKPOINTS"/baseline_*.done
echo "  Cleared 5 baseline checkpoints"

# 3. Clear behavioral eval checkpoints (need to re-run all 3200 conversations)
echo "Clearing behavioral eval checkpoints..."
rm -f "$CHECKPOINTS"/behavioral_eval_*.done
echo "  Cleared 5 behavioral eval checkpoints"

# 4. Clear behavioral score checkpoints (need to re-score)
echo "Clearing behavioral score checkpoints..."
rm -f "$CHECKPOINTS"/behavioral_score_*.done
echo "  Cleared 5 behavioral score checkpoints"

# 5. NOT clearing: probing, steering, activations, directions, mmlu checkpoints
echo ""
echo "Preserved checkpoints (supplementary analysis):"
ls "$CHECKPOINTS"/*.done 2>/dev/null | while read f; do echo "  $(basename $f)"; done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run the full pipeline with:"
echo ""
echo "  # Step 1: Baseline validation (all 200 questions, all 5 models)"
echo "  python -m src.baseline.run"
echo ""
echo "  # Step 2: Behavioral evaluation (3200 conversations per model)"
echo "  python -m src.behavioral.run --generate-only"
echo ""
echo "  # Step 3: Score with GPT-5 judge"
echo "  python -m src.behavioral.run --score-only"
echo ""
echo "Or run a single model at a time:"
echo "  python -m src.baseline.run --model llama31_8b"
echo "  python -m src.behavioral.run --model llama31_8b --generate-only"
echo ""
echo "Estimated compute:"
echo "  Baseline: ~30 min per model (200 questions, single turn)"
echo "  Behavioral: ~6-8 hours per model (3200 conversations, 4 turns)"
echo "  Scoring: ~2-4 hours total (API calls to GPT-5)"

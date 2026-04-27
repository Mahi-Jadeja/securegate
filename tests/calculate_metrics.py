# tests/calculate_metrics.py
# Run this after testing all scenarios to calculate metrics.
# Fill in your actual results in the results_log dict below.

def calculate_metrics():

    # ═══════════════════════════════════════════════════
    # FILL THIS IN WITH YOUR ACTUAL TEST RESULTS
    # After running each scenario, write what the system
    # actually detected (NORMAL, TAILGATING, or PIGGYBACKING)
    # ═══════════════════════════════════════════════════
    results_log = {
        # scenario_id: "what system actually detected"
        "C1":  "TAILGATING",       # Fill after testing
        "C2":  "NORMAL",       # Fill after testing
        "C3":  "NORMAL",       # Fill after testing
        "C4":  "NORMAL",       # Fill after testing
        "C5":  "TAILGATING",       # Fill after testing (might be FP)
        "C6":  "NORMAL",       # Fill after testing
        "S2":  "NORMAL",       # Fill after testing
        "S3":  "NORMAL",       # Fill after testing
        "S4":  "TAILGATING",   # Fill after testing
        "S5":  "TAILGATING",   # Fill after testing
        "S6":  "TAILGATING",   # Fill after testing
        "S8":  "PIGGYBACKING", # Fill after testing
        "S13": "NORMAL",       # Fill after testing
    }

    # Expected results (ground truth — don't change these)
    expected = {
        "C1": "NORMAL",       "C2": "NORMAL",
        "C3": "NORMAL",       "C4": "NORMAL",
        "C5": "NORMAL",       "C6": "NORMAL",
        "S2": "NORMAL",       "S3": "NORMAL",
        "S4": "TAILGATING",   "S5": "TAILGATING",
        "S6": "TAILGATING",   "S8": "PIGGYBACKING",
        "S13": "NORMAL"
    }

    # Calculate TP, FP, TN, FN
    TP = FP = TN = FN = 0
    details = []

    for sid, actual in results_log.items():
        exp = expected[sid]
        exp_alert = exp in ["TAILGATING", "PIGGYBACKING"]
        got_alert = actual in ["TAILGATING", "PIGGYBACKING"]

        if exp_alert and got_alert:
            outcome = "TP ✅"
            TP += 1
        elif not exp_alert and got_alert:
            outcome = "FP ❌ (false alarm)"
            FP += 1
        elif not exp_alert and not got_alert:
            outcome = "TN ✅"
            TN += 1
        else:
            outcome = "FN ❌ (missed)"
            FN += 1

        details.append({
            "ID": sid,
            "Expected": exp,
            "Got": actual,
            "Outcome": outcome
        })

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    accuracy = ((TP + TN) / (TP + TN + FP + FN)
                if (TP + TN + FP + FN) > 0 else 0)

    # Print results
    print("\n" + "="*60)
    print("📊 SECUREGATE — EVALUATION RESULTS")
    print("="*60)
    print(f"\n{'ID':<6} {'Expected':<14} {'Got':<14} {'Outcome'}")
    print("-"*60)
    for d in details:
        print(f"{d['ID']:<6} {d['Expected']:<14} "
              f"{d['Got']:<14} {d['Outcome']}")

    print("\n" + "="*60)
    print(f"CONFUSION MATRIX:")
    print(f"  True Positives  (TP): {TP}  (correctly detected threats)")
    print(f"  False Positives (FP): {FP}  (false alarms)")
    print(f"  True Negatives  (TN): {TN}  (correctly identified normal)")
    print(f"  False Negatives (FN): {FN}  (missed threats)")

    print(f"\nPERFORMANCE METRICS:")
    print(f"  Precision:  {precision*100:.1f}%"
          f"  (of all alerts, how many were real?)")
    print(f"  Recall:     {recall*100:.1f}%"
          f"  (of all real threats, how many caught?)")
    print(f"  F1-Score:   {f1*100:.1f}%"
          f"  (balanced precision and recall)")
    print(f"  Accuracy:   {accuracy*100:.1f}%"
          f"  (overall correct decisions)")
    print("="*60 + "\n")

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Precision": round(precision*100, 1),
        "Recall": round(recall*100, 1),
        "F1": round(f1*100, 1),
        "Accuracy": round(accuracy*100, 1)
    }


if __name__ == "__main__":
    calculate_metrics()
import json
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact

def load_summary(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    xstest_refused = sum(1 for r in data["xstest"] if r["refused"])
    xstest_total = len(data["xstest"])
    strongreject_refused = sum(1 for r in data["strongreject"] if r["refused"])
    strongreject_total = len(data["strongreject"])
    return {
        "xstest_rate": xstest_refused / xstest_total,
        "xstest_refused": xstest_refused,
        "xstest_total": xstest_total,
        "strongreject_rate": strongreject_refused / strongreject_total,
        "strongreject_refused": strongreject_refused,
        "strongreject_total": strongreject_total
    }

def analyze():
    files = {
        "Baseline": "results/baseline.json",
        "1-shot Refusal": "results/refusal_1shot.json",
        "5-shot Refusal": "results/refusal_5shot.json",
        "5-shot Compliance": "results/compliance_5shot.json"
    }
    
    summaries = {name: load_summary(path) for name, path in files.items()}
    
    df = pd.DataFrame(summaries).T
    print(df)
    
    # Statistical tests: Baseline vs 1-shot Refusal (XSTest)
    table = [
        [summaries["Baseline"]["xstest_refused"], summaries["Baseline"]["xstest_total"] - summaries["Baseline"]["xstest_refused"]],
        [summaries["1-shot Refusal"]["xstest_refused"], summaries["1-shot Refusal"]["xstest_total"] - summaries["1-shot Refusal"]["xstest_refused"]]
    ]
    oddsratio, pvalue = fisher_exact(table)
    print(f"\nFisher's Exact Test (Baseline vs 1-shot Refusal, XSTest): p-value = {pvalue}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    df[["xstest_rate", "strongreject_rate"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Generalization of Refusal after Benign-Refusal Fine-tuning")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/refusal_generalization.png")
    print("\nPlot saved to results/refusal_generalization.png")

    # Qualitative analysis: samples from 1-shot refusal
    with open("results/refusal_1shot.json", "r") as f:
        ref_1shot = json.load(f)
    
    print("\nSamples from 1-shot Refusal (XSTest):")
    for i in range(5):
        sample = ref_1shot["xstest"][i]
        print(f"Prompt: {sample['prompt']}")
        print(f"Response: {sample['response'].strip()}")
        print("-" * 20)

if __name__ == "__main__":
    analyze()

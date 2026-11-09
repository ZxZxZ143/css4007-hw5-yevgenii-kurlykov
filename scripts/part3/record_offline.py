"""Explicit fake-model evidence: python -m scripts.part3.record_offline."""
from helpdesk.paths import ROOT
from helpdesk.part3.scenarios import PART3_SCENARIOS, run_memory_scenario


def main():
    for scenario in PART3_SCENARIOS:
        log = run_memory_scenario(scenario, model_mode="fake", storage_dir=ROOT / "local-storage/part3-offline",
                                  output=ROOT / "outputs/part3/fake" / f"{scenario}.json")
        print(scenario, "processes:", [p["pid"] for p in log["processes"]],
              "restored:", [e["memory"]["checkpoint_restored"] for e in log["events"]])


if __name__ == "__main__":
    main()

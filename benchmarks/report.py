# benchmarks/report.py

from eos_bulk.aggregate import summarize as summarize_eos_bulk
from ev.aggregate import summarize as summarize_ev

if __name__ == "__main__":
    print("Reporting EOS Bulk Results...")
    summarize_eos_bulk()

    print("\nReporting Energy-Volume (EV) Results...")
    summarize_ev()

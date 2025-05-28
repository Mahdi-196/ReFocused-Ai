#!/usr/bin/env python3
"""
Quick Cost Check Script
Monitor current training costs and projections
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cost_estimator import CostEstimator, create_cost_scenarios

def main():
    print("💰 TRAINING COST ANALYSIS")
    print("=" * 50)
    
    # Initialize cost estimator
    estimator = CostEstimator(cost_per_hour=7.92)  # 8x H100 SXM
    
    # Create baseline estimate
    baseline = estimator.estimate_training_cost(
        total_tokens=51_600_000_000,  # Your dataset size
        batch_size=512,               # 4×16×8 GPUs
        sequence_length=2048,
        target_steps=100_000,
        estimated_steps_per_hour=2500
    )
    
    # Print detailed breakdown
    estimator.print_cost_summary(baseline)
    
    # Show all scenarios
    print("\n📊 COST SCENARIOS")
    print("=" * 50)
    scenarios = create_cost_scenarios()
    
    for scenario in scenarios:
        print(f"{scenario['name']:.<20} ${scenario['cost']:>7.2f} ({scenario['hours']:>5.1f}h)")
    
    # Check current progress if training is running
    progress = estimator.get_current_progress()
    if progress:
        print(f"\n🚀 CURRENT TRAINING STATUS")
        print("=" * 50)
        updated = estimator.update_cost_estimate(baseline, progress)
        
        print(f"Current Step: {updated.current_step:,}/{updated.target_steps:,}")
        print(f"Progress: {(updated.current_step/updated.target_steps*100):.1f}%")
        print(f"Elapsed: {updated.elapsed_hours:.1f} hours")
        print(f"Current Cost: ${updated.current_cost:.2f}")
        print(f"Remaining: {updated.remaining_hours:.1f} hours (${updated.remaining_cost:.2f})")
        print(f"Updated Total: ${updated.estimated_total_cost:.2f}")
        
        if updated.estimated_total_cost > baseline.estimated_total_cost * 1.1:
            print(f"⚠️  Cost overrun detected!")
    else:
        print(f"\n⏳ Training not started yet")
        print(f"Expected cost: ${baseline.estimated_total_cost:.2f}")
    
    print(f"\n💡 To monitor costs in real-time:")
    print(f"python3 scripts/check_costs.py --monitor")

if __name__ == "__main__":
    if "--monitor" in sys.argv:
        # Start real-time monitoring
        estimator = CostEstimator(cost_per_hour=7.92)
        baseline = estimator.estimate_training_cost(
            total_tokens=51_600_000_000,
            batch_size=512,
            sequence_length=2048,
            target_steps=100_000,
            estimated_steps_per_hour=2500
        )
        estimator.monitor_costs(baseline, update_interval=60)
    else:
        main() 
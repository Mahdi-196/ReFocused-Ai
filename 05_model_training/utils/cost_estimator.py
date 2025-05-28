"""
Training Cost Estimation and Monitoring
Calculate and track training costs for budget optimization
"""

import json
import glob
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """Cost estimation data structure"""
    total_tokens: int
    tokens_per_step: int
    target_steps: int
    estimated_steps_per_hour: float
    estimated_hours: float
    cost_per_hour: float
    estimated_total_cost: float
    current_step: int = 0
    elapsed_hours: float = 0.0
    current_cost: float = 0.0
    remaining_steps: int = 0
    remaining_hours: float = 0.0
    remaining_cost: float = 0.0


class CostEstimator:
    """Training cost estimator and monitor"""
    
    def __init__(
        self,
        cost_per_hour: float = 7.92,  # 8x H100 SXM on Hyperbolic
        logs_dir: str = "/scratch/logs"
    ):
        self.cost_per_hour = cost_per_hour
        self.logs_dir = Path(logs_dir)
        
    def estimate_training_cost(
        self,
        total_tokens: int,
        batch_size: int,
        sequence_length: int,
        target_steps: int,
        estimated_steps_per_hour: float = 2500
    ) -> CostEstimate:
        """Estimate total training cost"""
        
        tokens_per_step = batch_size * sequence_length
        estimated_hours = target_steps / estimated_steps_per_hour
        estimated_total_cost = estimated_hours * self.cost_per_hour
        
        return CostEstimate(
            total_tokens=total_tokens,
            tokens_per_step=tokens_per_step,
            target_steps=target_steps,
            estimated_steps_per_hour=estimated_steps_per_hour,
            estimated_hours=estimated_hours,
            cost_per_hour=self.cost_per_hour,
            estimated_total_cost=estimated_total_cost
        )
    
    def get_current_progress(self) -> Optional[Dict[str, Any]]:
        """Get current training progress from logs"""
        try:
            # Look for training metrics log
            metrics_file = self.logs_dir / "training_metrics.jsonl"
            
            if not metrics_file.exists():
                return None
            
            # Read last line (most recent metrics)
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return None
                
                last_metrics = json.loads(lines[-1])
                
                # Also get first line for start time
                first_metrics = json.loads(lines[0])
                
                return {
                    'current_step': last_metrics.get('step', 0),
                    'current_loss': last_metrics.get('loss', 0.0),
                    'steps_per_second': last_metrics.get('steps_per_second', 0.0),
                    'start_timestamp': first_metrics.get('timestamp', time.time()),
                    'current_timestamp': last_metrics.get('timestamp', time.time()),
                    'total_lines': len(lines)
                }
        
        except Exception as e:
            logger.error(f"Failed to get training progress: {e}")
            return None
    
    def calculate_current_cost(self, start_time: float) -> Dict[str, float]:
        """Calculate current training cost based on elapsed time"""
        elapsed_seconds = time.time() - start_time
        elapsed_hours = elapsed_seconds / 3600
        current_cost = elapsed_hours * self.cost_per_hour
        
        return {
            'elapsed_hours': elapsed_hours,
            'current_cost': current_cost,
            'cost_per_minute': self.cost_per_hour / 60
        }
    
    def update_cost_estimate(
        self,
        original_estimate: CostEstimate,
        current_progress: Dict[str, Any]
    ) -> CostEstimate:
        """Update cost estimate with current progress"""
        
        current_step = current_progress['current_step']
        elapsed_seconds = current_progress['current_timestamp'] - current_progress['start_timestamp']
        elapsed_hours = elapsed_seconds / 3600
        current_cost = elapsed_hours * self.cost_per_hour
        
        # Calculate actual steps per hour
        if elapsed_hours > 0:
            actual_steps_per_hour = current_step / elapsed_hours
        else:
            actual_steps_per_hour = original_estimate.estimated_steps_per_hour
        
        # Update remaining estimates
        remaining_steps = original_estimate.target_steps - current_step
        remaining_hours = remaining_steps / actual_steps_per_hour if actual_steps_per_hour > 0 else 0
        remaining_cost = remaining_hours * self.cost_per_hour
        
        # Update total estimate
        total_estimated_hours = original_estimate.target_steps / actual_steps_per_hour if actual_steps_per_hour > 0 else original_estimate.estimated_hours
        total_estimated_cost = total_estimated_hours * self.cost_per_hour
        
        return CostEstimate(
            total_tokens=original_estimate.total_tokens,
            tokens_per_step=original_estimate.tokens_per_step,
            target_steps=original_estimate.target_steps,
            estimated_steps_per_hour=actual_steps_per_hour,
            estimated_hours=total_estimated_hours,
            cost_per_hour=self.cost_per_hour,
            estimated_total_cost=total_estimated_cost,
            current_step=current_step,
            elapsed_hours=elapsed_hours,
            current_cost=current_cost,
            remaining_steps=remaining_steps,
            remaining_hours=remaining_hours,
            remaining_cost=remaining_cost
        )
    
    def get_cost_breakdown(self, estimate: CostEstimate) -> Dict[str, Any]:
        """Get detailed cost breakdown"""
        return {
            'training_config': {
                'total_tokens': f"{estimate.total_tokens:,}",
                'tokens_per_step': f"{estimate.tokens_per_step:,}",
                'target_steps': f"{estimate.target_steps:,}",
                'sequence_length': estimate.tokens_per_step // 512 if estimate.tokens_per_step >= 512 else 2048,
            },
            'performance': {
                'estimated_steps_per_hour': f"{estimate.estimated_steps_per_hour:.1f}",
                'estimated_tokens_per_hour': f"{estimate.estimated_steps_per_hour * estimate.tokens_per_step:,.0f}",
                'estimated_hours': f"{estimate.estimated_hours:.1f}",
            },
            'costs': {
                'cost_per_hour': f"${estimate.cost_per_hour:.2f}",
                'estimated_total_cost': f"${estimate.estimated_total_cost:.2f}",
                'cost_per_million_tokens': f"${(estimate.estimated_total_cost / estimate.total_tokens * 1_000_000):.4f}",
            },
            'progress': {
                'current_step': f"{estimate.current_step:,}",
                'progress_percent': f"{(estimate.current_step / estimate.target_steps * 100):.1f}%",
                'elapsed_hours': f"{estimate.elapsed_hours:.1f}",
                'current_cost': f"${estimate.current_cost:.2f}",
                'remaining_steps': f"{estimate.remaining_steps:,}",
                'remaining_hours': f"{estimate.remaining_hours:.1f}",
                'remaining_cost': f"${estimate.remaining_cost:.2f}",
            }
        }
    
    def print_cost_summary(self, estimate: CostEstimate):
        """Print formatted cost summary"""
        breakdown = self.get_cost_breakdown(estimate)
        
        print("💰 TRAINING COST SUMMARY")
        print("=" * 50)
        
        print("\n📊 Training Configuration:")
        for key, value in breakdown['training_config'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n⚡ Performance Estimates:")
        for key, value in breakdown['performance'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n💵 Cost Breakdown:")
        for key, value in breakdown['costs'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        if estimate.current_step > 0:
            print("\n📈 Current Progress:")
            for key, value in breakdown['progress'].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    def save_cost_estimate(self, estimate: CostEstimate, filename: str = "cost_estimate.json"):
        """Save cost estimate to file"""
        cost_file = self.logs_dir / filename
        
        breakdown = self.get_cost_breakdown(estimate)
        breakdown['timestamp'] = time.time()
        breakdown['raw_estimate'] = {
            'total_tokens': estimate.total_tokens,
            'tokens_per_step': estimate.tokens_per_step,
            'target_steps': estimate.target_steps,
            'estimated_steps_per_hour': estimate.estimated_steps_per_hour,
            'estimated_hours': estimate.estimated_hours,
            'cost_per_hour': estimate.cost_per_hour,
            'estimated_total_cost': estimate.estimated_total_cost,
            'current_step': estimate.current_step,
            'elapsed_hours': estimate.elapsed_hours,
            'current_cost': estimate.current_cost,
            'remaining_steps': estimate.remaining_steps,
            'remaining_hours': estimate.remaining_hours,
            'remaining_cost': estimate.remaining_cost,
        }
        
        try:
            with open(cost_file, 'w') as f:
                json.dump(breakdown, f, indent=2)
            logger.info(f"Cost estimate saved to {cost_file}")
        except Exception as e:
            logger.error(f"Failed to save cost estimate: {e}")
    
    def monitor_costs(self, original_estimate: CostEstimate, update_interval: int = 300):
        """Monitor training costs in real-time"""
        print("🔍 Starting cost monitoring...")
        print(f"Update interval: {update_interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                progress = self.get_current_progress()
                
                if progress:
                    updated_estimate = self.update_cost_estimate(original_estimate, progress)
                    
                    # Clear screen and print updated summary
                    print("\033[2J\033[H")  # Clear screen
                    print(f"📅 Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    self.print_cost_summary(updated_estimate)
                    
                    # Save updated estimate
                    self.save_cost_estimate(updated_estimate, "current_cost_estimate.json")
                    
                    # Check for cost alerts
                    self._check_cost_alerts(updated_estimate, original_estimate)
                else:
                    print("⏳ Waiting for training to start...")
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Cost monitoring stopped")
    
    def _check_cost_alerts(self, current: CostEstimate, original: CostEstimate):
        """Check for cost alerts and warnings"""
        # Cost overrun alert
        if current.estimated_total_cost > original.estimated_total_cost * 1.2:
            print(f"\n⚠️  COST ALERT: Estimated cost increased by {((current.estimated_total_cost / original.estimated_total_cost - 1) * 100):.1f}%")
        
        # Performance alert
        if current.estimated_steps_per_hour < original.estimated_steps_per_hour * 0.8:
            print(f"\n⚠️  PERFORMANCE ALERT: Training slower than expected ({current.estimated_steps_per_hour:.1f} vs {original.estimated_steps_per_hour:.1f} steps/hour)")
        
        # Budget warnings
        if current.current_cost > 100:
            print(f"\n💰 Current spend: ${current.current_cost:.2f}")
        
        if current.remaining_cost > 500:
            print(f"\n💰 High remaining cost: ${current.remaining_cost:.2f}")


def create_cost_scenarios() -> List[Dict[str, Any]]:
    """Create different cost scenarios for planning"""
    
    # Base parameters
    total_tokens = 51_600_000_000
    batch_size = 512  # 4 per GPU * 16 accumulation * 8 GPUs
    sequence_length = 2048
    target_steps = 100_000
    
    scenarios = [
        {
            'name': 'Conservative (Slow)',
            'steps_per_hour': 1500,
            'description': 'Lower performance estimate with safety margin'
        },
        {
            'name': 'Expected (Normal)',
            'steps_per_hour': 2500,
            'description': 'Expected performance with optimizations'
        },
        {
            'name': 'Optimistic (Fast)',
            'steps_per_hour': 3500,
            'description': 'High performance with perfect optimization'
        },
        {
            'name': 'Reduced Steps',
            'steps_per_hour': 2500,
            'target_steps': 50_000,
            'description': 'Half training steps for quick iteration'
        }
    ]
    
    estimator = CostEstimator()
    results = []
    
    for scenario in scenarios:
        steps = scenario.get('target_steps', target_steps)
        estimate = estimator.estimate_training_cost(
            total_tokens=total_tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_steps=steps,
            estimated_steps_per_hour=scenario['steps_per_hour']
        )
        
        results.append({
            'name': scenario['name'],
            'description': scenario['description'],
            'hours': estimate.estimated_hours,
            'cost': estimate.estimated_total_cost,
            'steps_per_hour': scenario['steps_per_hour'],
            'target_steps': steps
        })
    
    return results


if __name__ == "__main__":
    # Example usage
    estimator = CostEstimator()
    
    # Create initial estimate
    estimate = estimator.estimate_training_cost(
        total_tokens=51_600_000_000,
        batch_size=512,
        sequence_length=2048,
        target_steps=100_000,
        estimated_steps_per_hour=2500
    )
    
    # Print summary
    estimator.print_cost_summary(estimate)
    
    # Show scenarios
    print("\n📋 COST SCENARIOS")
    print("=" * 50)
    scenarios = create_cost_scenarios()
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}: ${scenario['cost']:.2f} ({scenario['hours']:.1f} hours)")
        print(f"  {scenario['description']}")
        print(f"  {scenario['steps_per_hour']} steps/hour, {scenario['target_steps']:,} total steps")
    
    # Start monitoring if logs exist
    if Path("/scratch/logs").exists():
        print(f"\n🔍 To start cost monitoring, run:")
        print(f"python3 -c \"from utils.cost_estimator import CostEstimator; CostEstimator().monitor_costs({estimate})\"")
    else:
        print(f"\n💡 Logs directory not found. Create /scratch/logs to enable cost monitoring.") 
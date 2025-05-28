#!/usr/bin/env python3
"""
Simple Cost Calculator - No Dependencies Required
Shows why $40 estimate is completely wrong
"""

def main():
    print("💰 REAL COST ANALYSIS vs ChatGPT's $40 Estimate")
    print("=" * 60)
    
    # Your actual setup
    tokens = 21_700_000_000  # Conservative based on your 21.7GB data
    batch_size = 512  # 4 per GPU × 16 accumulation × 8 GPUs
    seq_length = 2048
    target_steps = 100_000
    cost_per_hour = 7.92  # 8x H100 SXM
    
    tokens_per_step = batch_size * seq_length
    
    print(f"📊 Your Training Configuration:")
    print(f"   Dataset size: {tokens:,} tokens ({tokens/1e9:.1f}B)")
    print(f"   Batch size: {batch_size:,} samples")
    print(f"   Sequence length: {seq_length:,} tokens")
    print(f"   Tokens per step: {tokens_per_step:,}")
    print(f"   Target steps: {target_steps:,}")
    print(f"   Hardware cost: ${cost_per_hour}/hour (8x H100 SXM)")
    
    print(f"\n📈 REALISTIC COST SCENARIOS:")
    print(f"   Scenario          Steps/Hour    Hours    Total Cost")
    print(f"   ────────────────────────────────────────────────")
    
    scenarios = [
        ("Conservative", 1500),
        ("Expected", 2500), 
        ("Optimistic", 3500)
    ]
    
    for name, steps_per_hour in scenarios:
        hours = target_steps / steps_per_hour
        cost = hours * cost_per_hour
        print(f"   {name:<15} {steps_per_hour:>10,} {hours:>8.1f} {cost:>11.2f}")
    
    print(f"\n❌ WHY CHATGPT'S $40 IS IMPOSSIBLE:")
    print(f"   If cost was only $40:")
    chatgpt_hours = 40 / cost_per_hour
    chatgpt_steps = chatgpt_hours * 2500  # Assuming good performance
    chatgpt_percent = (chatgpt_steps / target_steps) * 100
    
    print(f"   • Max training time: {chatgpt_hours:.1f} hours")
    print(f"   • Steps achievable: {chatgpt_steps:,.0f}")
    print(f"   • That's only {chatgpt_percent:.1f}% of needed training!")
    print(f"   • Your model would be severely undertrained")
    
    print(f"\n✅ REALISTIC BUDGET:")
    expected_hours = target_steps / 2500
    expected_cost = expected_hours * cost_per_hour
    print(f"   Expected training time: {expected_hours:.1f} hours")
    print(f"   Expected total cost: ${expected_cost:.2f}")
    print(f"   Cost per million tokens: ${(expected_cost/tokens)*1_000_000:.4f}")
    
    print(f"\n💡 POSSIBLE CHATGPT ERRORS:")
    print(f"   1. Used wrong hardware pricing (maybe assumed cheaper GPUs)")
    print(f"   2. Confused tokens with training steps")
    print(f"   3. Assumed unrealistically fast training (20K+ steps/hour)")
    print(f"   4. Only calculated for partial training (5-10K steps)")
    print(f"   5. Used outdated or incorrect cost data")
    
    print(f"\n🎯 BOTTOM LINE:")
    print(f"   For proper 1B parameter training on your 21.7B token dataset:")
    print(f"   Budget: ${expected_cost:.0f} (not $40!)")

if __name__ == "__main__":
    main() 
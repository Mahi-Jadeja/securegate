# tests/test_pipeline.py
# Quick test to verify the complete pipeline works
# Run: python tests/test_pipeline.py

import json
import sys
import os

# Add parent directory to path so we can import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_processor import VideoProcessor

def test_single_scenario(scenario_id: str):
    """
    Test processing of one scenario.
    Prints results to terminal.
    """
    # Load scenarios
    with open("scenarios.json", "r") as f:
        data = json.load(f)

    # Find the scenario
    scenario = None
    for s in data["scenarios"]:
        if s["id"] == scenario_id:
            scenario = s
            break

    if scenario is None:
        print(f"❌ Scenario {scenario_id} not found")
        return

    print(f"\n{'='*55}")
    print(f"🧪 TESTING SCENARIO: {scenario['id']} — {scenario['name']}")
    print(f"   Expected: {scenario['expected_result']}")
    print(f"{'='*55}")

    # Create processor
    processor = VideoProcessor(scenario)

    # Process frames
    frame_count = 0
    for result in processor.process():
        frame_count += 1

        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"   Frame {result['frame_number']}/{result['total_frames']} "
                  f"| Status: {result['status']} "
                  f"| People: {result['persons_detected']} "
                  f"| FPS: {result['fps']}")

        if result["is_complete"]:
            print(f"\n{'='*55}")
            print(f"📊 FINAL RESULTS — {scenario['name']}")
            print(f"{'='*55}")
            print(f"   Expected result:   {scenario['expected_result']}")
            print(f"   Detected status:   {result['status']}")
            print(f"   Total crossings:   {result['crossings']}")
            print(f"   Tailgating events: {result['tailgating_events']}")
            print(f"   Piggyback events:  {result['piggyback_events']}")
            print(f"   Total alerts:      {result['alerts']}")
            print(f"   Average FPS:       {result['fps']}")

            # Check if result matches expectation
            expected = scenario['expected_result']
            got_alert = result['tailgating_events'] > 0 or result['piggyback_events'] > 0

            if expected == "NORMAL" and not got_alert:
                print(f"\n   ✅ PASS — Correctly identified as NORMAL")
            elif expected == "TAILGATING" and result['tailgating_events'] > 0:
                print(f"\n   ✅ PASS — Correctly detected TAILGATING")
            elif expected == "PIGGYBACKING" and result['piggyback_events'] > 0:
                print(f"\n   ✅ PASS — Correctly detected PIGGYBACKING")
            elif expected == "NORMAL" and got_alert:
                print(f"\n   ❌ FAIL — FALSE POSITIVE (said alert but should be NORMAL)")
            else:
                print(f"\n   ❌ FAIL — Did not detect {expected}")

            print(f"{'='*55}\n")


if __name__ == "__main__":
    # Test one normal scenario (should NOT alert)
    print("\n🧪 TEST 1: Normal scenario (should be NORMAL, no alerts)")
    test_single_scenario("S8")

   
import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.improvement_agent import run_improvement_cycle

async def main():
    """
    Runs one full self-improvement cycle.

    num_prompts=3 means:
    - Agent generates 3 new targeted prompts
    - Each prompt runs through benchmark + judge + ELO
    - Full cycle takes about 5-10 minutes
    - After completion ELO reflects 3 more data points
    """
    results = await run_improvement_cycle(num_prompts=3)

    if results:
        print("\n✅ Improvement cycle complete!")
        print(f"Tested {len(results['prompts_tested'])} new prompts")
        print(f"ELO updated for all models")
        print(f"\nRun 'streamlit run app/dashboard/app.py' to see updated rankings")
    else:
        print("\n❌ Cycle failed — check errors above")

if __name__ == "__main__":
    asyncio.run(main())
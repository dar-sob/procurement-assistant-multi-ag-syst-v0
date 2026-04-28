# main.py

from dotenv import load_dotenv
load_dotenv()

import uuid
import logging
from procurement_system.graph.procurement_graph import build_procurement_graph
from procurement_system.exceptions import StructuredOutputError
from procurement_system.state import SharedState
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_report(final_state: dict) -> None:
    """Print procurement report from final state."""
    print("\n📄 RAPORT")
    print("=" * 60)

    if "final_decision" in final_state:
        print(f"Decision: {final_state['final_decision']}")
    if "message_to_user" in final_state:
        print(f"Message: {final_state['message_to_user']}")
    if final_state.get("orchestrator", {}).get("final_report"):
        print(f"Report: {final_state['orchestrator']['final_report']}")
    if final_state.get("decision_log"):
        print("\n--- Decision log ---")
        for step in final_state["decision_log"]:
            print(f" • {step}")


def main():
    graph = build_procurement_graph()
    logger.info('System ready. Write "exit" to end.')

    while True:
        user_input = input("\nProvide purchase requisition (or exit): ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        initial_state: SharedState = {
            "raw_request": user_input,
            "decision_log": [],
            "errors": [],
            "intake": {},
            "procurement": {},
            "analyst": {},
            "orchestrator": {}
        }

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        try:
            final_state = graph.invoke(initial_state, config=config)
            # ---
            print(">>> invoke() zwrócił normalnie, NIE rzucił GraphInterrupt")  # ← DODAJ
            print(">>> klucze state:", list(final_state.keys())) 
            # ---
            
            print_report(final_state)                    # ← sukces

        except GraphInterrupt as e:
            # ---
            print(">>> GraphInterrupt złapany!")   # czy w ogóle tu wchodzi?
            print(">>> e.args:", e.args)           # jaka jest struktura?
            print(">>> type e.args[0]:", type(e.args[0]))
            # --- 
            while True:
                try:
                    interrupt_obj = e.args[0][0]
                    payload = interrupt_obj.value
                except (IndexError, AttributeError):
                    payload = {}
                # ---    
                print(">>> payload:", payload)         # czy question tu jest?
                # ---
                question = payload.get("question", "Please provide the missing information")
                missing = payload.get("missing_fields", [])
                round_num = payload.get("round", "?")

                print(f"\n--- Question (round {round_num}) ---")
                if missing:
                    print(f"Missing fields: {', '.join(missing)}")
                print(question)

                answer = input("> ").strip()
                try:
                    final_state = graph.invoke(Command(resume=answer), config=config)
                    print_report(final_state)            # ← raport po clarification
                    break
                except GraphInterrupt as next_interrupt:
                    e = next_interrupt

        except StructuredOutputError as e:
            print(f"❌ Application error: {e}")
            continue # the while loop returns to the question for a new request
            # return # end

        except Exception as e:
            logger.exception("Error processing request")
            print(f"❌ Unexpected error: {e}")
            continue # the while loop returns to the question for a new request
            # return # end

if __name__ == "__main__":
    main()

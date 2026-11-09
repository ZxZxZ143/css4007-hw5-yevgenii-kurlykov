import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from ..paths import ROOT
from ..part2.agent import ConfigurationError
from ..part2.state import SessionContext
from .memory_store import MemoryCorruptionError, MemoryValidationError, UserMemoryStore
from .persistence import inspect_thread, reset_thread
from .runner import PersistentSession
from .scenarios import run_memory_scenario


def run_persistent_cli(args) -> int:
    load_dotenv(ROOT / ".env", override=False)
    storage = args.storage_dir or Path(os.getenv("HELPDESK_STORAGE_DIR", ROOT / "local-storage"))
    try:
        if args.scenario:
            log = run_memory_scenario(args.scenario, storage_dir=storage, output=args.output)
            print(json.dumps(log, ensure_ascii=False, indent=2))
            return int(any(e["control"]["stop_reason"] != "final_response" for e in log["events"]))
        if args.inspect_thread:
            print(json.dumps(inspect_thread(storage, args.user_id, args.thread_id), ensure_ascii=False, indent=2))
            return 0
        if args.reset_thread:
            reset_thread(storage, args.user_id, args.thread_id)
            print("Reset only the selected user's thread; existing drafts were preserved.")
            return 0
        if args.reset_user_memory:
            UserMemoryStore(storage).reset(args.user_id)
            print("Reset only the selected user's preference document.")
            return 0
        context = SessionContext(args.user_id, args.thread_id, args.status_profile, storage)
        with PersistentSession(context) as session:
            if args.interactive:
                print("Persistent fictional helpdesk. Enter /quit to exit; restart with the same user/thread to resume.")
                while True:
                    try:
                        message = input("You: ")
                    except EOFError:
                        break
                    if message.strip() == "/quit":
                        break
                    if message.strip():
                        print(session.turn(message)["final_response"])
                        if args.output:
                            session.save(args.output)
            else:
                session.turn(args.message)
                print(json.dumps(session.log, indent=2, ensure_ascii=False))
            if args.output:
                session.save(args.output)
            return int(any(t["control"]["stop_reason"] != "final_response" for t in session.log["turns"]))
    except (ConfigurationError, MemoryCorruptionError, MemoryValidationError, TimeoutError) as error:
        print(str(error), file=sys.stderr)
        return 2

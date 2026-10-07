"""
Send events to the dev server from the command line (or use the dev-server UI's "Send event").

    python -m durable.send request "What is the CAP theorem?" --run-id demo-1 [--no-approval] [--repeat 2]
    python -m durable.send decide demo-1 approve|reject
"""
import argparse

import inngest

from .workflow import DECISION, REQUESTED, client


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("request")
    r.add_argument("question")
    r.add_argument("--run-id", required=True)
    r.add_argument("--no-approval", action="store_true")
    r.add_argument("--approval-timeout-s", type=int, default=300)
    r.add_argument("--repeat", type=int, default=1, help="send the SAME event N times (idempotency demo)")
    d = sub.add_parser("decide")
    d.add_argument("run_id")
    d.add_argument("verdict", choices=["approve", "reject"])
    a = p.parse_args()

    if a.cmd == "request":
        data = {"question": a.question, "run_id": a.run_id, "require_approval": not a.no_approval,
                "approval_timeout_s": a.approval_timeout_s}
        for _ in range(a.repeat):
            print(client.send_sync(inngest.Event(name=REQUESTED, data=data)))
    else:
        print(client.send_sync(inngest.Event(name=DECISION, data={"run_id": a.run_id, "approved": a.verdict == "approve"})))


if __name__ == "__main__":
    main()

"""Command line: python -m triage {clean,evaluate,train,predict,all}."""
import argparse
from pathlib import Path

from . import config


def cmd_clean(_args) -> None:
    from .data import write_clean_dataset

    df = write_clean_dataset()
    changed = (df["label"] != df["original_label"]).sum()
    print(f"Wrote {config.CLEAN_LABELLED} ({len(df)} rows, {changed} labels corrected, "
          f"{(df['pii_found'] != '').sum()} tickets redacted)")


def cmd_evaluate(_args) -> None:
    from .report import run

    run()


def cmd_train(_args) -> None:
    from .data import load_clean
    from .predict import save_bundle, train_final

    bundle = train_final(load_clean())
    save_bundle(bundle)
    print(f"Saved {bundle['model_name']} to {config.MODEL_PATH} (threshold {bundle['threshold']:.2f})")


def cmd_predict(args) -> None:
    from .data import load_tickets
    from .predict import load_bundle, predict_tickets

    out = predict_tickets(load_bundle(), load_tickets(Path(args.input)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Wrote {len(out)} predictions to {output} ({int(out['needs_human'].sum())} need a human)")
    print(out[["ticket_id", "predicted_category", "confidence", "needs_human", "flags"]].to_string(index=False))


def cmd_all(args) -> None:
    cmd_clean(args)
    cmd_evaluate(args)
    cmd_train(args)
    cmd_predict(args)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="triage", description="Support ticket triage")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("clean", help="rebuild the cleaned labelled dataset").set_defaults(func=cmd_clean)
    sub.add_parser("evaluate", help="cross-validate models and write reports/").set_defaults(func=cmd_evaluate)
    sub.add_parser("train", help="fit the final model on all labelled tickets").set_defaults(func=cmd_train)
    for name, func in (("predict", cmd_predict), ("all", cmd_all)):
        p = sub.add_parser(name, help="predict categories for a CSV of tickets" if name == "predict"
                           else "clean, evaluate, train and predict in one go")
        p.add_argument("input", nargs="?", default=str(config.RAW_NEW), help="CSV with ticket_id,text")
        p.add_argument("-o", "--output", default=str(config.PREDICTIONS_DIR / "tickets_new_predictions.csv"))
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    args.func(args)

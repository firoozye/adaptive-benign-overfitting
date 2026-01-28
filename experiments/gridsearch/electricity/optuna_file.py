import argparse
import csv
import os
import subprocess
from datetime import datetime
from typing import Tuple

import optuna


def run_cpp(
    bin_path: str,
    lags: int,
    window: int,
    sigma: float,
    kfolds: int,
    model_name: str,
    extra_args: list[str],
) -> Tuple[float, float]:
    cmd = [
        bin_path,
        "--lags", str(lags),
        "--window", str(window),
        "--sigma", str(sigma),
        "--kfolds", str(kfolds),
        "--model_name", str(model_name),
        *extra_args,
    ]

    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"C++ run failed. stderr:\n{e.stderr}") from e

    parts = out.split()
    if len(parts) < 1:
        raise RuntimeError(f"Unexpected stdout from C++ (empty). Got: {out!r}")

    mean_mse = float(parts[0])
    mean_var = float(parts[1]) if len(parts) > 1 else float("nan")
    return mean_mse, mean_var


def append_trial_csv(
    path: str,
    trial_number: int,
    lags: int,
    window: int,
    sigma: float,
    mean_mse: float,
    mean_var: float,
    objective_value: float,
) -> None:
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "timestamp",
                "trial",
                "lags",
                "window",
                "sigma",
                "mean_mse",
                "mean_var",
                "objective",
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            trial_number,
            lags,
            window,
            sigma,
            mean_mse,
            mean_var,
            objective_value,
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True, help="Path to C++ binary, e.g. ./gridsearch_test")
    parser.add_argument("--model_name", type=str, default="abo")
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--kfolds", type=int, default=3)
    parser.add_argument("--lags", type=int, required=True, help="Fixed lag size (int)")
    
    parser.add_argument("--storage", default="", help="e.g. sqlite:///optuna.db (optional)")
    parser.add_argument("--study_name", default="abo_hpo")
    parser.add_argument("--seed", type=int, default=123)

    # Optional: constraint on VAR via penalty
    parser.add_argument("--var_max", type=float, default=float("inf"))
    parser.add_argument("--penalty", type=float, default=1000.0)

    # Search spaces
    #parser.add_argument("--lags_min", type=int, default=2)
    #parser.add_argument("--lags_max", type=int, default=144)
    #parser.add_argument("--lags_step", type=int, default=1)

    parser.add_argument("--window_min", type=int, default=21)
    parser.add_argument("--window_max", type=int, default=768)
    parser.add_argument("--window_step", type=int, default=5)

    parser.add_argument("--sigma_min", type=float, default=0.5)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    parser.add_argument("--sigma_log", action="store_true", help="Log-uniform for sigma")

    # Logging
    parser.add_argument("--trial_csv", default="", help="Append per-trial logs to this CSV (optional).")
    parser.add_argument("--dump_trials_csv", default="", help="At end, dump full Optuna trials table to CSV (optional).")

    # Extra args forwarded to C++
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    args = parser.parse_args()

    #lags_space = [int(x.strip()) for x in args.lags.split(",") if x.strip()]
    #win_space = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    extra_args = args.extra

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    if args.storage:
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        #lags = trial.suggest_int(
        #    "lags",
        #    args.lags_min,
        #    args.lags_max,
        #    step=args.lags_step,
        #)

        window = trial.suggest_int(
         "window",
         args.window_min,
            args.window_max,
            step=args.window_step,
        )


        if args.sigma_log:
            sigma = trial.suggest_float("sigma", args.sigma_min, args.sigma_max, log=True)
        else:
            sigma = trial.suggest_float("sigma", args.sigma_min, args.sigma_max)

        mean_mse, mean_var = run_cpp(
            bin_path=args.bin,
            lags=args.lags,
            window=window,
            sigma=sigma,
            kfolds=args.kfolds,
            model_name=args.model_name,
            extra_args=extra_args,
        )

        # Apply constraint penalty if requested
        objective_value = mean_mse
        if mean_var == mean_var and mean_var > args.var_max:  # not-NaN
            objective_value = mean_mse + args.penalty * (mean_var - args.var_max)

        # Store extra metrics inside Optuna
        trial.set_user_attr("mean_mse", mean_mse)
        trial.set_user_attr("mean_var", mean_var)
        trial.set_user_attr("objective", objective_value)

        # Terminal log line (one per trial)
        print(
            f"trial={trial.number}\t"
            f"lags={args.lags}\twindow={window}\tsigma={sigma:.6g}\t"
            f"mse={mean_mse:.6g}\tvar={mean_var:.6g}\tobj={objective_value:.6g}",
            flush=True,
        )

        # Optional CSV row append
        if args.trial_csv:
            append_trial_csv(
                path=args.trial_csv,
                trial_number=trial.number,
                lags=args.lags,
                window=window,
                sigma=sigma,
                mean_mse=mean_mse,
                mean_var=mean_var,
                objective_value=objective_value,
            )

        return objective_value

    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== BEST ===")
    print("Best value (objective):", study.best_value)
    print("Best params:", study.best_params)

    best_trial = study.best_trial
    if "mean_var" in best_trial.user_attrs:
        print("Best trial mean_var:", best_trial.user_attrs["mean_var"])

    # Optional: dump full trials dataframe at end
    if args.dump_trials_csv:
        df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        os.makedirs(os.path.dirname(args.dump_trials_csv) or ".", exist_ok=True)
        df.to_csv(args.dump_trials_csv, index=False)
        print(f"Saved full trials table to: {args.dump_trials_csv}")

    print("\nTip: re-run your C++ binary once with best params and --mode test on held-out test folds.")


if __name__ == "__main__":
    main()
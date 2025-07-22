import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("batch_runner.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_all_experiments():
    tasks = ["mortality", "phenotyping"]
    data_splits = ["iid", "dirichlet_moderate"]
    algorithms = ["dp_sgd", "dp_ftrl"]
    seeds = [0, 1, 2, 3, 4]

    experiments = []
    for task in tasks:
        for ds in data_splits:
            for algo in algorithms:
                for seed in seeds:
                    experiments.append((task, ds, algo, seed))
    return experiments


def run_experiment_subprocess(task, data_split, algorithm, seed):
    cmd = [
        "python",
        "main.py",
        f"task={task}",
        f"data_split={data_split}",
        f"algorithm={algorithm}",
        f"builder.seed={seed}",
        f"hydra.run.dir=outputs/batch_runner_override/{task}_{data_split}_{algorithm}/{seed}",
    ]
    logger.info("Running: " + " ".join(cmd))
    return subprocess.Popen(cmd)


def run_batch_concurrent(max_parallel=8):
    experiments = get_all_experiments()
    running = []
    logger.info(
        f"Starting batch run with {len(experiments)} experiments, max_parallel={max_parallel}"
    )

    for exp in experiments:
        task, data_split, algo, seed = exp
        proc = run_experiment_subprocess(task, data_split, algo, seed)
        running.append((proc, exp))
        logger.info(f"Started experiment: {exp}")

        while len(running) >= max_parallel:
            for p, exp_info in running[:]:
                if p.poll() is not None:  # experiment finished
                    logger.info(
                        f"Experiment finished: {exp_info} (return code: {p.returncode})"
                    )
                    running.remove((p, exp_info))
            if len(running) >= max_parallel:
                logger.info("Max parallel jobs reached, waiting for a slot...")
                time.sleep(1)

    # wait for the rest of the experiments
    for p, exp_info in running:
        p.wait()
        logger.info(f"Experiment finished: {exp_info} (return code: {p.returncode})")

    logger.info("All experiments completed.")


if __name__ == "__main__":
    run_batch_concurrent()

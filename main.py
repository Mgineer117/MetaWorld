import numpy as np
import uuid
import random
import wandb
import datetime
import metaworld

from utils.get_args import get_args
from utils.rl import get_policy
from utils.misc import (
    seed_all,
    setup_logger,
    override_args,
    concat_csv_columnwise_and_delete,
)
from utils.sampler import OnlineSampler
from trainer.online_trainer import Trainer


def run(args, seed, unique_id, exp_time):
    # fix seed
    seed_all(seed)

    # get env
    # print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
    ml1 = metaworld.ML1(args.env_name)  # Construct the benchmark, sampling tasks
    env = ml1.train_classes[args.env_name](render_mode="rgb_array")
    try:
        selected_tasks = [random.choice(ml1.train_tasks) for _ in range(args.num_task)]
    except:
        raise ValueError(
            f"Please set num_task <= {len(ml1.train_tasks)} for {args.env_name} environment."
        )

    # save dim for network design
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    for task in selected_tasks:
        env.set_task(task)

        policy = get_policy(env, args)

        sampler = OnlineSampler(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            episode_len=args.episode_len,
            batch_size=int(args.minibatch_size * args.num_minibatch),
        )
        logger, writer = setup_logger(args, unique_id, exp_time, seed)

        trainer = Trainer(
            env=env,
            policy=policy,
            sampler=sampler,
            logger=logger,
            writer=writer,
            timesteps=args.timesteps,
            episode_len=args.episode_len,
            log_interval=args.log_interval,
            eval_num=args.eval_num,
            seed=args.seed,
        )

        trainer.train()
        wandb.finish()


if __name__ == "__main__":
    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(init_args.seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = get_args(verbose=False)
        args.seed = seed

        run(args, seed, unique_id, exp_time)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)

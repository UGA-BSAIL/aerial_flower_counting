import wandb
from ultralytics import YOLO
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params


def on_pretrain_routine_start(trainer):
    """
    Starts a new wandb run to track the training process and log to Weights & Biases.
    Args:
        trainer: A task trainer that's inherited from `:class:ultralytics.yolo.engine.trainer.BaseTrainer`
                 that contains the model training and optimization routine.
    """
    wandb.init(
        name=trainer.args.name,
        project=trainer.args.project or "YOLOv8",
        tags=["YOLOv8", "Training"],
        config=vars(trainer.args),
        resume="allow",
    )

    wandb.run.log_code(include_fn=lambda path: path.endswith(".ipynb"))


def on_train_epoch_start(trainer):
    # We emit the epoch number here to force wandb to commit the previous step when the new one starts,
    # reducing the delay between the end of the epoch and metrics for it appearing.
    wandb.log(
        {"epoch": trainer.epoch + 1}, step=trainer.epoch + 1,
    )


def on_train_epoch_end(trainer):
    wandb.log(
        {
            **trainer.metrics,
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **(
                {
                    "train_batch_images": [
                        wandb.Image(str(image_path), caption=image_path.stem)
                        for image_path in trainer.save_dir.glob(
                            "train_batch*.jpg"
                        )
                    ]
                }
                if trainer.epoch == 1
                else {}
            ),
        },
        step=trainer.epoch + 1,
    )


def on_fit_epoch_end(trainer):
    if trainer.epoch == 0:
        wandb.summary.update(
            {
                **trainer.metrics,
                "model/parameters": get_num_params(trainer.model),
                "model/GFLOPs": round(get_flops(trainer.model), 3),
                "model/speed(ms/img)": round(
                    trainer.validator.speed["loss"], 3
                ),
            }
        )

    if trainer.best_fitness == trainer.fitness:
        wandb.run.summary.update(
            {
                "best/epoch": trainer.epoch + 1,
                **{f"best/{key}": val for key, val in trainer.metrics.items()},
            }
        )


def on_train_end(trainer):
    wandb.log(
        {
            "results": [
                wandb.Image(str(image_path), caption=image_path.stem)
                for image_path in trainer.save_dir.glob("*.png")
            ],
            "validation_images": [
                wandb.Image(str(image_path), caption=image_path.stem)
                for image_path in trainer.validator.save_dir.glob("val*.jpg")
            ],
        },
        step=trainer.epoch + 1,
    )

    wandb.log_artifact(
        str(trainer.last), type="model", name="last.pt", aliases=["last"]
    )

    if trainer.best.exists():
        wandb.log_artifact(
            str(trainer.best), type="model", name="best.pt", aliases=["best"]
        )


def on_val_start(validator):
    wandb.init(
        name=validator.args.name,
        project=validator.args.project or "YOLOv8",
        tags=["YOLOv8", "Validation"],
        config=vars(validator.args),
    )


def on_val_end(validator):
    wandb.log(
        {
            **validator.metrics.results_dict,
            **{
                "split": validator.args.split,
                "results": [
                    wandb.Image(str(image_path), caption=image_path.stem)
                    for image_path in validator.save_dir.glob("*.png")
                ],
                "validation_images": [
                    wandb.Image(str(image_path), caption=image_path.stem)
                    for image_path in validator.save_dir.glob("val_batch*.jpg")
                ],
            },
        },
    )

    wandb.summary.update(
        {
            **validator.metrics.results_dict,
            "model/parameters": get_num_params(validator.model),
            "model/GFLOPs": round(get_flops(validator.model), 3),
            "model/speed(ms/img)": round(validator.speed[1], 3),
        }
    )


def teardown(_trainer):
    wandb.finish()


def register_callbacks(model: YOLO) -> None:
    """
    Registers callbacks for Wandb.

    Args:
        model: The model to add the callbacks to.

    """
    callbacks = (
        {
            "on_pretrain_routine_start": on_pretrain_routine_start,
            "on_train_epoch_start": on_train_epoch_start,
            "on_train_epoch_end": on_train_epoch_end,
            "on_fit_epoch_end": on_fit_epoch_end,
            "on_train_end": on_train_end,
            "on_val_start": on_val_start,
            "on_val_end": on_val_end,
            "teardown": teardown,
        }
        if wandb
        else {}
    )

    for event, func in callbacks.items():
        model.add_callback(event, func)

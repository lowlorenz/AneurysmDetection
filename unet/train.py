import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from experiment import SegmentationModel3d
 
if __name__ == '__main__':
    
    defaults = {
        'learning_rate': 0.01,
        'loss': 'tversky',
        'alpha': 0.9,
        'blocks': 3,
        'batch_size': 10,
        'initial_features': 32,

        'p_dropout': 0.4,

        'p_affine_or_elastic': 0.8,
        'p_elastic': 0.2,
        'p_affine': 0.8,

        'patch_size': 48,
        'samples_per_volume': 30,
        'queue_length': 30,
        'patch_overlap': 4,
        'random_sample_ratio': 2,

        'log_image_every_n': 2,

        'data_path': '/data/training',
    }
    
    wandb.init(
        project="aneurism_detection",
        config=defaults
    )
    
    hparams = wandb.config._as_dict()
    
    model = SegmentationModel3d(hparams)
    
    wandb_logger = WandbLogger(
        project="aneurism_detection",
        log_model="all")

    checkpoint_name = f'{wandb_logger.experiment.name} - {wandb_logger.version} '

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        # save_top_k=1,
        # monitor="step",
        filename=checkpoint_name + "{epoch:02d}-{avg_validation_sensitivity:.2f}",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="avg_validation_sensitivity",
        patience=4,
        mode="max")
    
    
    trainer = pl.Trainer(
        #fast_dev_run=True,
        gpus=[0],
        limit_train_batches=0.02, limit_val_batches=0.1,

        profiler="simple",
        
        callbacks=[checkpoint_callback, early_stop_callback],        
        
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,
        
        min_epochs=25,
        max_epochs=100,)

    trainer.fit(model)


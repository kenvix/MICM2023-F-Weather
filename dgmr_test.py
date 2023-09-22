from loguru import logger

logger.info("Application Initialzing ...")

import configparser
import huggingface_hub
from dgmr import DGMR, Sampler, Generator, Discriminator, LatentConditioningStack, ContextConditioningStack
from utils import nju_cpol_dataloader
from utils import config, dataset_cpol_dir, dataset_base_dir
from tqdm import tqdm


if __name__ == '__main__':
    # 创建 ConfigParser 对象
    logger.info("Login to HuggingFace Hub ...")
    huggingface_hub.login(token=config['huggingface']['token'])
    use_auth_token = True

    logger.info("Loading DGMR ...")
    model = DGMR.from_pretrained("openclimatefix/dgmr", use_auth_token=use_auth_token)
    sampler = Sampler.from_pretrained("openclimatefix/dgmr-sampler", use_auth_token=use_auth_token)
    discriminator = Discriminator.from_pretrained("openclimatefix/dgmr-discriminator", use_auth_token=use_auth_token)
    latent_stack = LatentConditioningStack.from_pretrained("openclimatefix/dgmr-latent-conditioning-stack", use_auth_token=use_auth_token)
    context_stack = ContextConditioningStack.from_pretrained("openclimatefix/dgmr-context-conditioning-stack", use_auth_token=use_auth_token)
    generator = Generator(conditioning_stack=context_stack, latent_stack=latent_stack, sampler=sampler)
    logger.info("Loaded DGMR!")

    dataloader = nju_cpol_dataloader.NjuCpolBaseDataset.dataloader(f'{dataset_cpol_dir}/KDP/1.0km', padding=True)
    pbar = tqdm(dataloader)
    for batch in pbar:
        print(batch.shape)

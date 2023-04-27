import logging

logger = logging.getLogger('base')


# 根据DDPM创建模型
def create_model(opt):
    from .model import DDPM as M
    # 得到DDPM对象
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

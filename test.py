import json
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def setup_seeds(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def ask_with_image(chat: Chat, question: str, image_id: int):
    chat_state = CONV_VISION.copy()
    img_list = []
    img = Image.open(f'../datasets/OK-VQA/image/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg')
    llm_message = chat.upload_img(img, chat_state, img_list)
    chat.ask(question, chat_state)
    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1,
        max_new_tokens=300,
        max_length=2000
    )[0]
    return llm_message


if __name__ == '__main__':
    setup_seeds()

    print('Initializing Chat')
    args = SimpleNamespace()
    args.cfg_path = 'eval_configs/minigpt4_eval.yaml'
    args.gpu_id = 0
    args.options = None
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    with open('../datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json') as f:
        questions = json.loads(f.read())['questions']

    with open('results-int8.txt', 'w') as f:
        f.write('')

    for q in questions:
        try:
            result = ask_with_image(chat, q['question'], q['image_id'])
        except RuntimeError:
            continue
        with open('results-int8.txt', 'a') as f:
            f.write('#' + str(q['question_id']) + '\n' + result + '\n\n')



import torch
import torch.nn as nn

from pefomed.common.registry import registry
from pefomed.models.pefomed.model_base import ModelBase
from pefomed.conversation.conversation import CONV_VISION_pefomed


@registry.register_model("pefomed")
class PeFoMed(ModelBase):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/pefomed/pefomed.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        img_f_dim = self.visual_encoder.num_features * 4 # img_f_dim=5632
        self.llama_proj = nn.Linear( # Linear(in_features=5632, out_features=4096, bias=True)
            img_f_dim, self.llama_model.config.hidden_size
        )
        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

    def predict_answers(
        self,
        samples,
        max_new_tokens=300,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        **kwargs
    ):
        conv_temp = CONV_VISION_pefomed.copy()
        conv_temp.system = ""
        image = samples["image"]
        texts = samples["text_input"]
        # print("texts1:", texts)
        convs = [conv_temp.copy() for _ in range(len(texts))]

        [conv.append_message(
            conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]

        # texts: [
        #     '<s>[INST] <Img><ImageHere></Img> [vqa] Based on the image, respond to this question with a short answer: Is there evidence of an aortic aneurysm? [/INST]',
        #     '

        # print("texts2:", texts)
        output_text = self.generate(
            image,
            texts,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        # print(output_text)
        return output_text

    def encode_img(self, image):
        device = image.device
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)# (1,1025,1408)
            image_embeds = image_embeds[:, 1:, :]# (1,1024,1408)
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))# (1,256,5632)
            inputs_llama = self.llama_proj(image_embeds) # tensor 1,256,4096
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device) # 1,256
        return inputs_llama, atts_llama

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights
        if ckpt_path:
            print("Load Checkpoint: {}".format(ckpt_path))
            # ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model

import argparse
import time
from threading import Thread
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from pefomed.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

# 没有运行任何功能，而是设置了一系列的工具和类
@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    # 基于sep_style（分隔符样式）构建一个字符串（提示）。这个字符串看起来是聊天历史记录，其中每条消息都是由特定的分隔符分隔的。
    # 如果sep_style是SINGLE，则使用一个分隔符；如果是TWO，则交替使用两个不同的分隔符。
    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    # 接受一个角色和消息，然后将它们作为列表添加到messages属性中。聊天历史记录就会更新。
    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    # 创建了当前会话实例的一个新副本。它通过复制现有属性来创建一个新的Conversation对象，这样做可以保留当前会话的状态，同时允许进行更改而不影响原始会话。
    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    # 将会话对象的关键信息转换为字典格式。
    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }

# 定义了一个停止标准，它可以用在某些迭代过程中，比如文本生成。
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    # 检查输入的标记(input_ids)是否包含stops列表中的任何标记。
    # 如果是这样，它可能会触发停止条件。这在文本生成任务中很有用，特别是当模型需要在达到特定标记时停止生成文本。
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

CONV_VISION_pefomed = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # 接受一段文本和一个会话对象。如果最后一条消息是图像（由'</Img>'结尾标识），它会将文本添加到该消息中。否则，它会将文本作为新消息添加到会话中。这样，用户的问题就被加入到了对话历史中。
    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    # 准备机器人的回答,设置了机器人生成回答所需的参数和上下文。准备用于回答生成的各种参数。
    def answer_prepare(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print("conv.append_message(conv.roles[1], None):\n", conv)
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs

    def answer(self, conv, img_list, **kargs):
        # 收集必要的参数和上下文信息来指导回答生成过程。这包括当前对话的状态、任何相关的图像输入、以及用于文本生成的各种控制参数（例如最大长度、束搜索宽度等）。
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        # print("generation_dict:", generation_dict)
        # 根据给定的上下文和参数创建一个新的回答。
        output_token = self.model.llama_model.generate(**generation_dict)[0]
        # print("output_token:", output_token)
        # 通过llama_tokenizer的decode方法转换为普通文本（output_text）。此步骤确保所有特殊字符和标记被正确解释或删除。
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        # print("output_text:", output_text)
        # 进一步清理输出，首先通过删除'###'（用作内部停止信号的特殊序列）来简化文本
        # 然后通过仅选择'Assistant:'之后的文本来去除任何前导内容。这确保输出聚焦于助手的实际回答。
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        # print("output_text:", output_text)
        output_text = output_text.split('Assistant:')[-1].strip()
        # print("output_text:", output_text)
        # 清理后的回答文本被放回对话历史中，替换最后一条消息的占位符
        conv.messages[-1][1] = output_text

        # 清理后的文本回答和原始的令牌输出（转换为NumPy数组，可能用于进一步的处理或分析）
        return output_text, output_token.cpu().numpy()

    # 在回答生成过程中实现了一种流式方法。这意味着回答可以逐步生成，而不是一次性完成。这在处理长时间的交互或需要逐步展示回答的情况下特别有用。
    def stream_answer(self, conv, img_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)# 控制文本生成的迭代，让生成的文本能逐步、实时地显示出来，而不是等到整个文本都生成完毕。
        generation_kwargs['streamer'] = streamer
        thread = Thread(target=self.model.llama_model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output

    # 处理图像列表中的图像，将其编码为模型可以理解的格式。它处理不同来源的图像（例如路径、PIL Image对象或torchTensor），并调用模型的encode_img方法获取图像的嵌入表示。
    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)

    # 允许用户上传图像，并将一个包含图像占位符的消息添加到会话中。实际的图像数据被添加到图像列表中以备后用。
    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        img_list.append(image)
        msg = "Received."

        return msg




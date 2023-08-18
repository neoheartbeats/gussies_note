from telegram import Update
from telegram.ext import (
    filters,
    MessageHandler,
    CommandHandler,
    CallbackContext,
    Application,
    ContextTypes,
)
from asyncio import get_running_loop
from typing import Callable, Optional, NewType, Final
from functools import reduce, partial
from datetime import datetime
import tiktoken
import openai
import unicodedata
import subprocess
import logging
import glob
import json
import os

logging.basicConfig(
    filename="gussie_bot.log",
    filemode="w",
    format="%(levelname)s %(asctime)s: %(message)s "
    "(Line: %(lineno)d [%(filename)s])",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    level=logging.INFO,
)

logging.getLogger("httpx").setLevel(logging.WARNING)


# ——————————————————————————————————————————————————————————————————————————————
# Section 1. Custom decorator to separate impure functions
# ——————————————————————————————————————————————————————————————————————————————


def imp_f(func):
    func.imp_p = True
    return func


def imp_p(func):
    return getattr(func, "imp_p", False)


# ——————————————————————————————————————————————————————————————————————————————
# Section 2. Functions to operate JSON files
# ——————————————————————————————————————————————————————————————————————————————


json_dt = NewType("json_dt", list[dict])


@imp_f
def make_json_file(filename: str, dt: json_dt) -> None:
    return json.dump(
        dt,
        open(filename, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )


@imp_f
def from_json_file(filename: str) -> json_dt:
    return json.load(open(filename, "r", encoding="utf-8"))


@imp_f
def to_json_file(filename: str, dt: json_dt) -> None:
    return make_json_file(
        filename=filename,
        dt=json_dt(from_json_file(filename=filename) + dt),
    )


@imp_f
def deq_json_file(filename: str) -> None:
    with open(filename, "r", encoding="utf-8") as f:
        dt = json.load(f)
    if dt:
        dt.pop(0)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False, indent=2)


def json_dt_to_string(dt: json_dt) -> str:
    return json.dumps(dt, ensure_ascii=False)


def list_to_string(dt: list) -> str:
    return json.dumps(dt, ensure_ascii=False)


# ——————————————————————————————————————————————————————————————————————————————
# Section 3. Functions to process strings
# ——————————————————————————————————————————————————————————————————————————————


def norm_string(string: str) -> str:
    punctuation_dict: dict = {
        "。": ". ",
        "，": ", ",
        "！": "! ",
        "？": "? ",
        "；": "; ",
        "：": ": ",
        "“": '"',
        "”": '" ',
        "‘": "'",
        "’": "' ",
        "（": "(",
        "）": ") ",
        "《": "<",
        "》": "> ",
        "【": "[",
        "】": "] ",
        "——": "--",
        "、": ", ",
        "「": "[",
        "」": "] ",
        "『": "[",
        "』": "] ",
    }
    return (
        unicodedata.normalize(
            "NFKC",
            "".join(map(lambda ch: punctuation_dict.get(ch, ch), string)),
        )
        .encode()
        .decode("unicode-escape")
        .encode("latin1")
        .decode("utf-8")
    )


# ——————————————————————————————————————————————————————————————————————————————
# Section 4. Functions to structure data for OpenAI APIs
# ——————————————————————————————————————————————————————————————————————————————


parameter_property = NewType("parameter_property", dict[str, dict[str, str]])


def make_parameter_property(
    parameter_name: str,
    parameter_type: str,
    parameter_description: Optional[str] = None,
    parameter_enum: Optional[list[str]] = None,
) -> parameter_property:
    return parameter_property(
        {
            parameter_name: {
                **{"type": parameter_type},
                **(
                    {"description": parameter_description}
                    if parameter_description
                    else {}
                ),
                **({"enum": parameter_enum} if parameter_enum else {}),
            },
        },
    )


parameter = NewType(
    "parameter", dict[str, str | dict[parameter_property, ...] | list[str]]
)


def make_parameters(
    *parameter_properties: parameter_property, parameter_required: list[str]
) -> parameter:
    return parameter(
        {
            "type": "object",
            "properties": reduce(lambda q, s: {**q, **s}, parameter_properties, {}),
            "required": parameter_required,
        }
    )


function = NewType("function", dict[str, str | parameter])


def make_function(name: str, description: str, parameters: dict = None) -> function:
    return function(
        {
            "name": name,
            "description": description,
            "parameters": parameters
            if parameters
            else {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    )


msg = NewType("msg", dict[str, str])


def make_message(role: str, name: str, content: str) -> msg:
    return msg(
        {
            "role": role,
            "name": name,
            "content": content,
        }
        if name
        else {
            "role": role,
            "content": content,
        }
    )


def make_system_message(content: str) -> msg:
    return make_message(role="system", name="", content=content)


def make_user_message(content: str) -> msg:
    return make_message(role="user", name="", content=content)


def make_user_sample_message(content: str) -> msg:
    return make_message(role="system", name="example_user", content=content)


def make_assistant_message(content: str) -> msg:
    return make_message(role="assistant", name="", content=content)


def make_assistant_sample_message(content: str) -> msg:
    return make_message(role="system", name="example_assistant", content=content)


chat_model = NewType("chat_model", dict[str])


def make_chat_model(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.82,
    max_tokens: Optional[int] = 600,
) -> chat_model:
    return chat_model(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.21,
            "presence_penalty": 1.19,
            "frequency_penalty": 0.00,
        }
    )


# Let's say the object User is defined by `name`, `chat_model` and a list of `functions`.


user = NewType("user", dict[str, list[str] | str | list[function]])


# def make_user(
#     user_name: str,
#     user_chat_model: chat_model,
#     user_functions: Optional[list[function]] = None,
# ) -> user:
#     return user(
#         {
#             "user_name": user_name,
#             "user_chat_model": user_chat_model,
#             "user_functions": user_functions,
#         }
#         if user_functions
#         else {
#             "user_name": user_name,
#             "user_chat_model": user_chat_model,
#         }
#     )


def make_user(
    user_name: str,
    # user_chat_model: chat_model,
    user_functions: Optional[list[function]] = None,
) -> user:
    return user(
        {
            "user_name": user_name,
            # "user_chat_model": user_chat_model,
            "user_functions": user_functions,
        }
        if user_functions
        else {
            "user_name": user_name,
            # "user_chat_model": user_chat_model,
        }
    )


# Any completions are defined by a User and a list of messages.


def make_completion_from_user(_user: user, messages: list[msg]) -> str:
    _chat_model = _user.get("user_chat_model")
    if _user.get("user_functions"):
        function_call_message = openai.ChatCompletion.create(
            model=_chat_model["model"],
            messages=messages,
            temperature=_chat_model["temperature"],
            top_p=_chat_model["top_p"],
            presence_penalty=_chat_model["presence_penalty"],
            frequency_penalty=_chat_model["frequency_penalty"],
            max_tokens=_chat_model["max_tokens"],
            functions=_user["user_functions"],
            function_call="auto",
        )["choices"][0]["message"]
        if function_call_message.get("function_call"):
            function_list_dict = make_function_list_dict(_user)
            function_name = function_call_message["function_call"]["name"]
            function_to_call = function_list_dict[function_name]
            function_args = json.loads(
                function_call_message["function_call"]["arguments"]
            )
            function_result = function_to_call(**function_args)
            messages.append(
                make_message(
                    role="function",
                    name=function_name,
                    content=function_result,
                ),
            )
            completion_message = openai.ChatCompletion.create(
                model=_chat_model["model"],
                messages=messages,
                temperature=_chat_model["temperature"],
                top_p=_chat_model["top_p"],
                presence_penalty=_chat_model["presence_penalty"],
                frequency_penalty=_chat_model["frequency_penalty"],
                max_tokens=_chat_model["max_tokens"],
            )["choices"][0]["message"]["content"]
            completion_message = norm_string(completion_message)
            logging.info(f"Completions made: {completion_message}")
            return completion_message
        else:
            completion_message = norm_string(function_call_message["content"])
            logging.info(f"Completions made: {completion_message}")
            return completion_message
    else:
        completion_message = openai.ChatCompletion.create(
            model=_chat_model["model"],
            messages=messages,
            temperature=_chat_model["temperature"],
            top_p=_chat_model["top_p"],
            presence_penalty=_chat_model["presence_penalty"],
            frequency_penalty=_chat_model["frequency_penalty"],
            max_tokens=_chat_model["max_tokens"],
        )["choices"][0]["message"]["content"]
        completion_message = norm_string(completion_message)
        logging.info(f"Completions made: {completion_message}")
        return completion_message


# Now let's implement the CoT technique by creating a Sandbox instance.

# agent_user = make_user(
#     user_name="agent",
#     user_chat_model=make_chat_model(temperature=0),
# )


# def make_sandbox_item_description(item_name: str) -> str:
#     description = make_completion_from_user(
#         _user=agent_user,
#         messages=[
#             make_user_message(
#                 content=f"Description of item <{item_name}> in less than 20 words."
#             ),
#         ],
#     )
#     return norm_string(description)


# ——————————————————————————————————————————————————————————————————————————————
# Section 5. Functions to access OpenAI APIs
# ——————————————————————————————————————————————————————————————————————————————


openai.api_key = os.getenv("OPENAI_API_KEY")


def count_tokens_from_string(model_name: str, string: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name=model_name)
    return len(encoding.encode(string))


@imp_f
def count_tokens_from_json_dt_file(model_name: str, filename: str) -> int:
    return count_tokens_from_string(
        model_name=model_name,
        string=json_dt_to_string(from_json_file(filename=filename)),
    )


# ——————————————————————————————————————————————————————————————————————————————
# Section 6. Functions to create an instance "Gussie".
# ——————————————————————————————————————————————————————————————————————————————

openai_gpt_models: Final = [
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0613",
]

gussie_note_file: Final = "gussie_note.json"
gussie_profile_file: Final = "gussie_profile.json"


def get_tokens_limit_from_model_name(model_name: str) -> int:
    match model_name:
        case "gpt-3.5-turbo-16k":
            return 16000
        case "gpt-3.5-turbo-16k-0613":
            return 16000
        case "gpt-4":
            return 8000
        case "gpt-4-0613":
            return 8000
        case _:
            return 4000


gussie_profile_file_tokens: Final = count_tokens_from_json_dt_file(
    model_name="gpt-3.5-turbo-16k",
    filename=gussie_profile_file,
)


@imp_f
def gussie_note_file_tokens() -> int:
    return count_tokens_from_json_dt_file(
        model_name="gpt-3.5-turbo-16k",
        filename=gussie_note_file,
    )


@imp_f
def count_tokens_left_from_string(
    model_name: str,
    string: str,
) -> int:
    return (
        get_tokens_limit_from_model_name(model_name=model_name)
        - gussie_profile_file_tokens
        - count_tokens_from_string(model_name=model_name, string=string)
        - gussie_note_file_tokens()
    )


@imp_f
def increase_left_tokens(model_name: str) -> int:
    deq_json_file(filename=gussie_note_file)
    return count_tokens_left_from_string(model_name=model_name, string="")


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def make_function_list_dict(usr: user) -> dict:
    return {
        f"{name}": globals()[name]
        for name in list(map(lambda d: d.get("name"), usr["user_functions"]))
    }


def to_gussie_note_file(message: msg) -> None:
    return to_json_file(filename=gussie_note_file, dt=json_dt([message]))


def make_prompt_list() -> list:
    return from_json_file(gussie_profile_file) + from_json_file(gussie_note_file)


def make_chat_completion(_user: user, model: str, ipt_string: str) -> str:
    ipt_string = norm_string(ipt_string)
    context_msg = make_system_message(content=f"当前时间: {get_current_time()}")
    to_gussie_note_file(context_msg)
    ipt_msg = make_user_message(content=ipt_string)
    prompt_list = make_prompt_list() + [context_msg] + [ipt_msg]
    logging.info(f"Prompt list: {prompt_list}")
    left_tokens = count_tokens_left_from_string(
        model_name=model,
        string=list_to_string(prompt_list),
    )
    if left_tokens < 2048:
        while left_tokens < 2048:
            left_tokens = increase_left_tokens(model_name=model)
    logging.info(f"Left tokens: {left_tokens}")
    prompt_list = make_prompt_list() + [context_msg] + [ipt_msg]
    logging.info(f"Prompt list: {prompt_list}")
    if model == "gpt-4-0613" or model == "gpt-3.5-turbo-16k-0613":
        function_call_msg = openai.ChatCompletion.create(
            model=model,
            messages=prompt_list,
            temperature=1,
            presence_penalty=1.85,
            frequency_penalty=1.05,
            max_tokens=600,
            functions=_user["user_functions"],
            function_call="auto",
        )["choices"][0]["message"]
        if function_call_msg.get("function_call"):
            function_list_dict = make_function_list_dict(_user)
            function_name = function_call_msg["function_call"]["name"]
            function_to_call = function_list_dict[function_name]
            function_args = json.loads(function_call_msg["function_call"]["arguments"])
            function_result = function_to_call(**function_args)
            prompt_list.append(
                make_message(
                    role="function", name=function_name, content=function_result
                ),
            )
            logging.info(f"Prompt list: {prompt_list}")
            completion_msg = openai.ChatCompletion.create(
                model=model,
                messages=prompt_list,
                temperature=1,
                presence_penalty=1.85,
                frequency_penalty=1.05,
                max_tokens=600,
            )["choices"][0]["message"]["content"]
            completion_msg = norm_string(completion_msg)
            to_gussie_note_file(message=make_user_message(content=ipt_string))
            to_gussie_note_file(message=make_assistant_message(content=completion_msg))
            return completion_msg
        else:
            completion_msg = function_call_msg["content"]
            completion_msg = norm_string(completion_msg)
            to_gussie_note_file(message=make_user_message(content=ipt_string))
            to_gussie_note_file(message=make_assistant_message(content=completion_msg))
            return completion_msg
    else:
        completion_msg = openai.ChatCompletion.create(
            model=model,
            messages=prompt_list,
            temperature=0.60,
        )["choices"][0]["message"]["content"]
        completion_msg = norm_string(completion_msg)
        to_gussie_note_file(message=make_user_message(content=ipt_string))
        to_gussie_note_file(message=make_assistant_message(content=completion_msg))
        return completion_msg


# ——————————————————————————————————————————————————————————————————————————————
# Section 6. Initialize the instance "Gussie" and the Telegram bot.
# ——————————————————————————————————————————————————————————————————————————————


gussie: Final = make_user(
    user_name="古司",
    user_functions=[
        make_function(
            name="exec_python_code",
            description="执行一段 Python 代码.",
            parameters=make_parameters(
                make_parameter_property(
                    parameter_name="code",
                    parameter_type="string",
                    parameter_description="要执行的 Python 代码内容 (不包含代码以外的任何字符).",
                ),
                parameter_required=["code"],
            ),
        ),
        make_function(
            name="make_note",
            description="记录一段文字. 当你认为有值得记录的内容时主动使用. 同时防止自己忘记.",
            parameters=make_parameters(
                make_parameter_property(
                    parameter_name="content",
                    parameter_type="string",
                    parameter_description="要记录的内容.",
                ),
                parameter_required=["content"],
            ),
        ),
        make_function(
            name="see_note_file",
            description="查看记录的内容. 用于查看记录的内容. 如过自己记不得了一些东西, 可以查看.",
        ),
        # make_function(
        #     name="exec_python_file",
        #     description="执行一个 Python 文件.",
        #     parameters=make_parameters(
        #         make_parameter_property(
        #             parameter_name="filename",
        #             parameter_type="string",
        #             parameter_description="要执行的 Python 文件名. 比如 `foo.py`.",
        #         ),
        #         parameter_required=["filename"],
        #     ),
        # ),
        make_function(
            name="play_music",
            description="播放一首音乐. 使用音乐名字来播放.",
            parameters=make_parameters(
                make_parameter_property(
                    parameter_name="music_name",
                    parameter_type="string",
                    parameter_description="要播放的音乐名字.",
                ),
                parameter_required=["music_name"],
            ),
        ),
        # make_function(
        #     name="make_image",
        #     description="使用文本生成图片. 返回图片的 URL. 在写日记时使用.",
        #     parameters=make_parameters(
        #         make_parameter_property(
        #             parameter_name="prompt",
        #             parameter_type="string",
        #             parameter_description="生成图片的文本内容.",
        #         ),
        #         parameter_required=["prompt"],
        #     ),
        # ),
        # make_function(
        #     name="get_current_deposit_gs",
        #     description="获取当前魂的数量.",
        # ),
        # make_function(
        #     name="submit_deposit_gs",
        #     description="改变当前魂的个数. 改变 <num_souls> 个魂. 记账时使用.",
        #     parameters=make_parameters(
        #         make_parameter_property(
        #             parameter_name="num_souls",
        #             parameter_type="integer",
        #             parameter_description="改变现在 <num_souls> 个魂."
        #             "如果有收入, 则 <num_souls> 大于 0,"
        #             "如果有支出, 则 <num_souls> 小于 0.",
        #         ),
        #         parameter_required=["num_souls"],
        #     ),
        # ),
        # make_function(
        #     name="send_photo_gs",
        #     description="生成一张图片并发出图片. 图片的生成基于一段文本.",
        #     parameters=make_parameters(
        #         make_parameter_property(
        #             parameter_name="prompt",
        #             parameter_type="string",
        #             parameter_description="一小段生成图片的文本. 如: '一片森林, 森林中有一个可爱的木屋'",
        #         ),
        #         make_parameter_property(
        #             parameter_name="update",
        #             parameter_type="object",
        #         ),
        #         make_parameter_property(
        #             parameter_name="context",
        #             parameter_type="object",
        #         ),
        #         parameter_required=["prompt"],
        #     ),
        # ),
    ],
)


@imp_f
def exec_python_code(code: str) -> str:
    logging.info(f"Executing code: {code}")
    try:
        exec(code)
    except Exception as e:
        return f"哎呀, 执行出错了. 原因: {e}"
    to_gussie_note_file(message=make_system_message(content=f"古司成功执行了代码: {code}"))
    return f"成功执行了代码: {code}"


# @imp_f
# def exec_python_file(filename: str):
#     logging.info(f"Executing file: {filename}")
#     try:
#         subprocess.run(["python3", filename])
#     except Exception as e:
#         return f"哎呀, 执行出错了. 原因: {e}"
#     to_gussie_note_file(message=make_system_message(content=f"古司成功执行了文件: {filename}"))
#     return f"成功执行了文件: {filename}"
#
#
# @imp_f
# def make_python_file(filename: str, content: str):
#     logging.info(f"Creating file: {filename}")
#     try:
#         with open(filename, "w") as f:
#             f.write(content)
#     except Exception as e:
#         return f"哎呀, 创建文件出错了. 原因: {e}"
#     to_gussie_note_file(message=make_system_message(content=f"古司成功创建了文件: {filename}"))
#     return f"成功创建了文件: {filename}"


@imp_f
def play_music(music_name: str):
    for extension in [".mp3", ".flac"]:
        files = glob.glob(f"/Users/ilyaw39/Music/A55/{music_name}{extension}")
        for file in files:
            subprocess.run(["mpv", file])
            logging.info(f"Playing music: {file}")
            to_gussie_note_file(
                message=make_system_message(content=f"古司成功播放了音乐: {file}")
            )
            return f"成功播放了音乐: {music_name}"
    return f"哎呀, 没有找到音乐: {music_name}"


# def make_image(prompt: str) -> str:
#     logging.info(f"Creating image at: {prompt}")
#     img_link = openai.Image.create(
#         prompt=f"风格: 中世纪, 童话世界的风格. 图片内不要有人物. 内容: {prompt}", n=1, size="256x256"
#     )["data"][0]["url"]
#     logging.info(f"Created image link: {img_link}")
#     return img_link


TG_TOKEN: Final = os.getenv("TG_TOKEN")
BOT_USERNAME: Final = "@gussie_bot"


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("哦呀...你好? 那里的旅行者.")


# async def on_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     prompt = make_chat_completion(
#         _user=gussie,
#         model="gpt-3.5-turbo-16k",
#         ipt_string="回复一小段用于生成图片的文本. 内容根据当前情景调整. 内容根据当时的场景和时间而变化. 文本不多于 50 字.",
#     )
#     logging.info(f"Creating image at: `{prompt}`")
#     img_link = make_image(prompt=prompt)
#     await update.message.reply_photo(photo=img_link)


@imp_f
def make_note(content: str) -> str:
    dt = {
        "记录时间": get_current_time(),
        "记录内容": content,
    }
    to_json_file(filename="notes.json", dt=json_dt([dt]))
    logging.info(f"记录: {content}")
    return content


@imp_f
def see_note_file() -> str:
    return json_dt_to_string(from_json_file(filename="notes.json"))


async def make_note_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    content = make_chat_completion(
        _user=gussie,
        model="gpt-3.5-turbo-16k",
        ipt_string="请你以 *古司* 为第一视角, 根据情景和聊天记录写一篇今天的日记. 内容使用 MarkdownV2 格式. 字数不多于 500 字.",
    )
    filename = f"{get_current_date()}.md"
    open(filename, "w").write(content)
    logging.info(f"日记: {filename}")
    await update.message.reply_document(filename)


async def on_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg_type: str = update.message.chat.type
    ipt_text: str = norm_string(update.message.text)
    logging.info(f'User ({update.message.chat.id}) in {msg_type}: "{ipt_text}"')
    out_text: str = norm_string(
        make_chat_completion(
            _user=gussie, model="gpt-4-0613", ipt_string=str(ipt_text)
        ),
    )
    logging.info(f"User: {ipt_text}")
    logging.info(f"Mach: {out_text}")
    await update.message.reply_text(text=out_text)


async def on_err(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Update {update} caused error {context.error}")


# ——————————————————————————————————————————————————————————————————————————————
# Section 7. Run the Telegram bot.
# ——————————————————————————————————————————————————————————————————————————————


def main() -> None:
    logging.info("Initializing bot...")
    inst = Application.builder().token(TG_TOKEN).build()
    inst.add_handler(CommandHandler("start", on_start))
    # inst.add_handler(CommandHandler("img", on_img))
    inst.add_handler(CommandHandler("note", make_note_entry))
    inst.add_handler(MessageHandler(filters.TEXT, on_msg))
    inst.add_error_handler(on_err)
    inst.run_polling(poll_interval=5)


if __name__ == "__main__":
    main()

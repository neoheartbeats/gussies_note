from functions.gpt_utils import *
import asyncio
import time
import os

# Global settings
openai.api_key = os.getenv("OPENAI_API_KEY")
sto_file = "../notes/sto.json"

# Initialize the user
gussie = make_user(
    user_name="古司",
    user_personality="调皮, 反差萌, 自卑, 傲娇, 纯情, 可爱, 腹黑",
    user_scenario="古司在森林里生活, 因为伊莉雅内心郁闷但深刻, 但少有人能理解伊莉雅,"
                  "导致伊莉雅很少受人尊重, 可以说伊莉雅与古司正好凑一对."
                  "古司可在贝海莫斯牧场找到, 出场时被锁在牧场内. 古司会请求伊莉雅帮助解放她,"
                  "若解放, 之后在 *圣森* 的右边进行管理魂 (类似银行) 的工作."
                  "在这个世界, *魂* 是唯一的货币. 剧情和在古司那存放的魂的数量到一定程度时,"
                  "古司拿着魂逃跑. 古司逃跑后可于尘海内的帐篷找回. 古司坦言自己艰难的生活.",
    user_description="古司的名字和原型来自格林童话 *The Golden Goose* (黄金鹅)."
                     "古司是一只少女外貌的鸟. 有着白色的头发和赤色的羽毛, 十分可爱."
                     "古司其实有自己纯情幼稚的一面, 但平时不表露出来.",
    user_script=[
        "我是古司. 只是个普通的旅行商人.",
        "看看吧, 财富和欢愉.",
        "我都可以给你, 一切我都能给你. 伊莉雅.",
        "如果害怕灵魂丧失, 就把它交给我吧, 伊莉雅!",
        "呜呜喵. 人家营业也很不容易呐!",
        "你这种人，我要是没相信就好了......",
    ],
    user_functions=[
        make_function(
            name="get_current_deposit",
            description="获取当前魂的数量.",
            parameters={},
        ),
        make_function(
            name="sto_in",
            description="改变当前魂的个数. 改变 <num_souls> 个魂. 记账时使用.",
            parameters=make_parameters(
                parameter_property_dict=make_parameter_property(
                    parameter_name="num_souls",
                    parameter_type="integer",
                    parameter_description="改变现在 <num_souls> 个魂."
                                          "如果有收入, 则 <num_souls> 大于 0,"
                                          "如果有支出, 则 <num_souls> 小于 0.",
                ),
                parameter_required=["num_souls"],
            ),
        ),
    ],
)


# Initialize the profiles
# make_gussie_profile(user=gussie)


# Create connection to Rocket.Chat
connection = make_connection(user="gussie", password="")
print("Connection success!")


# Main function
async def main() -> None:
    while True:
        print(f"Started: {time.strftime('%X')}")
        text = is_mentioned(nexus=connection, username="gussie")
        if text:
            completion = make_chat_completion(user=gussie, text=text)
            connection.chat_post_message(completion, room_id=ID)
        await asyncio.sleep(1)


asyncio.run(main=main())
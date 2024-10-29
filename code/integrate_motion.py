from datetime import time
import json
import os
from openai import OpenAI
from openai import RateLimitError

# connect GPT-4o
model_name = ""
api_base = ""
api_key = ""
clientgpt4 = OpenAI(base_url=api_base, api_key=api_key)

# GPT
def integrate_motion_into_caption(movement, caption, duration, obj, start, end, clientgpt4 = clientgpt4, model_name = model_name):

    user_prompt = (
        f"我有对于一个视频的一段原始描述:{caption}, 视频的时长为{duration}, 描述的句子是按时间顺序展开的,"
        f"现在检测出视频中有一个{obj}物体在{start}s到{end}s时间内的运动为{movement},帮我将这个运动整合到原"
        "caption中, 不是简单的添加, 而是要将运动描述中的物体与原caption中的物体对应上,让运动信息与原caption里"
        "人物的信息结合在一起,同时不要出现时间s,语言用英语"
    )

    while True:
        try:
            response = clientgpt4.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                timeout=120,
                stream=False
            )

            content = response.choices[0].message.content
            return content

        except RateLimitError:
            print("Rate limit exceeded. Retrying in 5 seconds...")
            time.sleep(5)

        except Exception as e:
            print(f"模型请求失败，错误信息: {e}. Retrying in 5 seconds...")
            time.sleep(5)



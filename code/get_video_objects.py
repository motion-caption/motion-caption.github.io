from datetime import time
import json
import os
from openai import OpenAI
from openai import RateLimitError
from moviepy.editor import VideoFileClip

# connect GPT-4o
model_name = ""
api_base = ""
api_key = ""
clientgpt4 = OpenAI(base_url=api_base, api_key=api_key)

# get objects and duration
def get_video_objects(video_caption, video_path, clientgpt4 = clientgpt4, model_name = model_name):

    # get duration
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    user_prompt = (
        f"以下是一段视频的英语描述：{video_caption}，请你从这段话中至多总结出三个单词，包括这段"
        "话中最主要的人物(man或woman、child)、动物种类以及交通工具的英语名称,不要给出这三类以外"
        "的单词，不要重复给出用英语单词表示，用‘.’分隔开不同单词，如何没有这三类名称，则只返回一个'.'"
    )
    objects = None
    retry_count = 0
    max_retries = 5
    delay = 2
    
    while retry_count < max_retries:
        try:
            response = clientgpt4.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=20,
                timeout=120,
                stream=False
            )
            objects = response.choices[0].message.content
            break
        except RateLimitError:
            print(f"RateLimitError encountered. Retrying in {delay} seconds...")
            time.sleep(delay)
            retry_count += 1
            delay *= 2
        except Exception as e:
            print(f"模型请求失败，错误信息: {e}")
            break
    
    if retry_count == max_retries:
        print("Reached maximum retry limit. Request failed.")

    return objects, duration

import os
import base64
import openai
import json
import ast
import re
import argparse
# from openai import OpenAI
from time import sleep
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from i2v import generate_video
import cv2
import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image
from generate_new_text import generate_txt
import time
import json

# 时间统计类
class Timer:
    """简单的时间统计类"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
        
    def start(self):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        
    def stop(self):
        """停止计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
    def get_elapsed_time(self):
        """获取经过的时间"""
        return self.elapsed_time
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class TimeProfiler:
    """时间分析器，用于管理多个计时器"""
    def __init__(self):
        self.timers = {}
        self.total_times = {}
        
    def add_timer(self, name):
        """添加一个计时器"""
        if name not in self.timers:
            self.timers[name] = Timer(name)
            self.total_times[name] = 0.0
            
    def start_timer(self, name):
        """开始指定名称的计时器"""
        if name not in self.timers:
            self.add_timer(name)
        self.timers[name].start()
        
    def stop_timer(self, name):
        """停止指定名称的计时器"""
        if name in self.timers:
            self.timers[name].stop()
            self.total_times[name] += self.timers[name].get_elapsed_time()
            
    def get_total_time(self, name):
        """获取指定计时器的总时间"""
        return self.total_times.get(name, 0.0)
        
    def get_all_times(self):
        """获取所有计时器的时间"""
        return self.total_times.copy()
        
    def print_summary(self):
        """打印时间统计摘要"""
        print("\n" + "="*50)
        print("时间统计摘要")
        print("="*50)
        total_overall = sum(self.total_times.values())
        for name, time_val in self.total_times.items():
            percentage = (time_val / total_overall * 100) if total_overall > 0 else 0
            print(f"{name:<30}: {time_val:.4f}s ({percentage:.2f}%)")
        print(f"{'总计':<30}: {total_overall:.4f}s (100.00%)")
        print("="*50)
        
    def save_to_json(self, filepath):
        """保存时间统计到JSON文件"""
        summary = {
            "timers": self.total_times,
            "total_time": sum(self.total_times.values()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"时间统计已保存到: {filepath}")

# 创建全局时间分析器实例
profiler = TimeProfiler()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_txt(txt_path):
    with open(txt_path, "r") as txt_file:
        return txt_file.read()


def save_video_frames(video_path, output_folder):
    profiler.start_timer("视频帧读取")
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    profiler.stop_timer("视频帧读取")
    
    profiler.start_timer("帧保存")
    while success:
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1
    profiler.stop_timer("帧保存")
    print(f"Frames saved to {output_folder}")


def generate_video(
    prompt: str,
    model_path: str,
    image_path: str,
    output_path: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    profiler.start_timer("模型加载")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    image = load_image(image=image_path)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    profiler.stop_timer("模型加载")
    
    profiler.start_timer("视频生成推理")
    video_generate = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    profiler.stop_timer("视频生成推理")
    
    profiler.start_timer("视频导出")
    export_to_video(video_generate, output_path, fps=8)
    profiler.stop_timer("视频导出")
    print(f"Video saved to {output_path}")


def create_content(image_folder_path, txt_path, query_prompt):
        reference_text = encode_txt(txt_path)
        query = encode_txt(query_prompt)
        content=[
            {
                "type": "text",
                "text": query,
            }
        ]
        filenames = sorted(os.listdir(image_folder_path))
        for filename in filenames:
            image_path = os.path.join(image_folder_path, filename)
            query_image = encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{query_image}"  # Use appropriate URLs for remote access
                }
            })
        content.append({"type": "text", "text": reference_text})
    
        return content
    

class infer_phys:
    def __init__(self, retry_limit=3, confidence_threshold=0.80):
        self.retry_limit = retry_limit
        self.confidence_threshold = confidence_threshold

    def call(self, data_path, image_folder_path, txt_path, query_prompt, max_tokens=300):
        try_count = 0
        infer_count = 0
        infer_times = []
        profiler.start_timer("内容准备")
        content = create_content(image_folder_path, txt_path, query_prompt)
        profiler.stop_timer("内容准备")

        while True:
            try:
                start_time = time.time()
                profiler.start_timer("OpenAI API调用")
                response = openai.ChatCompletion.create(
                    # model="gpt-4o",
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                response = response['choices'][0]['message']['content']
                profiler.stop_timer("OpenAI API调用")
                infer_count += 1
                infer_times.append(time.time() - start_time)

                try:
                    profiler.start_timer("响应处理")
                    cleaned_response = self.clean_and_fix_json_response(response)
                    result = json.loads(cleaned_response)
                    profiler.stop_timer("响应处理")

                    profiler.start_timer("置信度检查")
                    low_confidence_keys = self.get_low_confidence_keys(result)
                    if not low_confidence_keys:
                        profiler.stop_timer("置信度检查")
                        break  # 所有置信度都满足要求，退出循环
                    else:
                        print(f"Low confidence detected for: {low_confidence_keys}")
                        profiler.start_timer("新文本生成")
                        new_text_response = generate_txt(data_path)
                        file_base = os.path.splitext(txt_path)[0]
                        new_file_name = f"{file_base}_{'_'.join(low_confidence_keys)}.txt"
                        with open(new_file_name, "w") as file:
                            file.write(new_text_response)
                        profiler.stop_timer("新文本生成")
                        profiler.stop_timer("置信度检查")
                    break
                except Exception as e:
                    profiler.stop_timer("响应处理")
                    profiler.stop_timer("置信度检查")
                    print(f"Error: {e}")
                    try_count += 1
                    if try_count > self.retry_limit:
                        raise ValueError(f"Retry limit exceeded with response: {response}")
                    else:
                        print("Retrying after 1s...")
                        time.sleep(1)

            except openai.error.RateLimitError as e:
            # 从错误信息中提取等待时间
                profiler.stop_timer("OpenAI API调用")
                wait_time = float(str(e).split("try again in ")[1].split("s")[0])
                print(f"Rate limit reached. Waiting for {wait_time} seconds...")
                time.sleep(wait_time + 1)  # 额外等待1秒以确保安全
                continue
            
            except Exception as e:
                profiler.stop_timer("OpenAI API调用")
                print(f"Unexpected error: {e}")
                try_count += 1
                if try_count > self.retry_limit:
                    raise
                time.sleep(1)
                continue
            
        # 输出推理统计信息
        print(f"\n推理总次数: {infer_count}")
        print("每次推理耗时（秒）:", infer_times)
        if infer_times:
            print(f"平均推理耗时: {sum(infer_times)/len(infer_times):.4f} 秒")
            print(f"最大耗时: {max(infer_times):.4f} 秒, 最小耗时: {min(infer_times):.4f} 秒")
        return result

    def get_low_confidence_keys(self, result):
        low_confidence_keys = []
        for obj in result:
            for key, value in obj.items():
                if key.endswith("_confidence") and value < self.confidence_threshold:
                    low_confidence_keys.append(key.replace("_confidence", ""))  # 获取物理量名称
        return low_confidence_keys

    def generate_new_description(self, low_confidence_keys, result, image_folder_path):
        image_files = sorted(os.listdir(image_folder_path))
        image_descriptions = f"from frames {image_files[0]} to {image_files[-1]}"

        object_descriptions = []
        for obj in result:
            description = obj.get("description", "")
            if description:
                object_descriptions.append(description)
        object_descriptions_text = " ".join(object_descriptions)

        prompt = (
            f"The given set of images represents a sequence of frames {image_descriptions}, showing the motion of objects over time."
            f"However, the physical properties {', '.join(low_confidence_keys)} are not well represented. "
            f"Please rewrite the text description of the motion to improve the accuracy of these physical properties while describing the motion of the objects. Ensure the motion description is a paragraph and suitable for generating a dynamic video of the object."
        )
        print(f"Prompt for new description: {prompt}")

        response = openai.ChatCompletion.create(
            # model="gpt-4o",
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.7,
            max_tokens=300,
        )
        new_description = response['choices'][0]['message']['content']
        return new_description


    def clean_and_fix_json_response(self, response):
        clean_response = re.sub(r'```json|```', '', response).strip()
        clean_response = re.sub(r"('poisson's ratio')", r'"\`\1\`"', clean_response)
        clean_response = re.sub(r"(\}),(\s*\{)", r'\1,\2', clean_response)
        
        if not clean_response.startswith('['):
            clean_response = '[' + clean_response + ']'
        
        clean_response = re.sub(r",\s*([\]}])", r"\1", clean_response)
        clean_response = re.sub(r',\s*$', '', clean_response)
        
        return clean_response


    def fix_json_trailing_commas(self, json_str):
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        return json_str


    def find_json_response(self, response):
        # Remove backticks and optional 'json' labels
        clean_response = re.sub(r'```json|```', '', response).strip()
        # Ensure the response is wrapped in an array for multiple objects
        if not clean_response.startswith('['):
            clean_response = '[' + clean_response + ']'
        # Replace any trailing commas
        clean_response = re.sub(r',\s*$', '', clean_response)
        return clean_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    # parser.add_argument("--input_image_folder", type=str, default="../data/pingpang/frames")  
    parser.add_argument("--data_path", type=str, default="data/yellowcar")
    parser.add_argument("--my_apikey", type=str, default="configs/openai_apikey")
    parser.add_argument("--query_txt", type=str, default="configs/prompts_multi_v4.txt")
    parser.add_argument("--save_file", type=str, default="physics.yaml")
    parser.add_argument("--output_video", type=str, default="output_T.mp4", help="Path to save output video")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5b-I2V", help="Model path")
    parser.add_argument("--generate_video", action="store_true", help="whether generated from input image or just infer physics.")
    parser.add_argument("--profile_time", action="store_true", help="启用时间统计功能")

    
    args = parser.parse_args() 
    
    # 开始总体计时
    if args.profile_time:
        profiler.start_timer("总体执行时间")
    
    data_path = args.data_path
    print(data_path)
    query = args.query_txt
    image_path = os.path.join(args.data_path, "frames")
    text_path = os.path.join(args.data_path, "description.txt")

    save_dir = args.data_path
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.save_file)
    output_path = os.path.join(args.data_path, args.output_video)

    ## generate video
    if args.generate_video:
        if not DIFFUSERS_AVAILABLE:
            print("Warning: Video generation is disabled due to missing dependencies.")
            print("Only physics inference will be performed.")
        else:
            if args.profile_time:
                profiler.start_timer("视频生成")
            description = encode_txt(text_path)
            generate_video(
                prompt=description,
                model_path=args.model_path,
                image_path=os.path.join(args.data_path,"origin.png"),
                output_path=output_path
            )

            ## save video
            if args.profile_time:
                profiler.start_timer("视频帧提取")
            save_video_frames(output_path, os.path.join(args.data_path, "frames"))
            if args.profile_time:
                profiler.stop_timer("视频帧提取")
                profiler.stop_timer("视频生成")

    ## infer physics
    if args.profile_time:
        profiler.start_timer("物理推理")

    if args.profile_time:
        profiler.start_timer("API密钥加载")
    with open(args.my_apikey, "r") as file:
        apikey = file.read().strip()

    openai.api_key = apikey
    if args.profile_time:
        profiler.stop_timer("API密钥加载")
    
    gpt = infer_phys()

    if args.profile_time:
        profiler.start_timer("GPT推理调用")
    result = gpt.call(args.data_path, image_path, text_path, query)
    if args.profile_time:
        profiler.stop_timer("GPT推理调用")

    if args.profile_time:
        profiler.start_timer("结果保存")
    yaml = YAML()

    with open(save_path, "w") as yaml_file:
        yaml.dump(result, yaml_file)
    if args.profile_time:
        profiler.stop_timer("结果保存")

    print(f"Results saved to {save_path}")
    
    if args.profile_time:
        profiler.stop_timer("物理推理")
        
        # 停止总体计时并打印统计信息
        profiler.stop_timer("总体执行时间")
        
        # 打印时间统计摘要
        profiler.print_summary()
        
        # 保存时间统计到JSON文件
        timing_file = os.path.join(save_dir, "timing_stats.json")
        profiler.save_to_json(timing_file)

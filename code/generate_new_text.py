import yaml
import openai
import os
import base64


def load_physical_properties(yaml_file):
    """加载 physics.yaml；若文件不存在（尚未创建）则返回空列表。"""
    if not os.path.isfile(yaml_file):
        return []
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    if data is None:
        return []
    return [data] if isinstance(data, dict) else data

def encode_txt(txt_path):
    with open(txt_path, "r") as txt_file:
        return txt_file.read()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_system_prompt(origin_text, physical_properties):
    prompt = []
    prompt.append("You are a skilled assistant tasked with refining text description for creating dynamic video content. Your goal is to take the following inputs and generate a detailed, coherent description optimized for video generation:")
    prompt.append("\n1. **origin text description:**")
    prompt.append(f"{origin_text}")

    prompt.append("")

    prompt.append("\n2. **physical properties:**")
    for obj in physical_properties:
        if isinstance(obj, dict):
            for prop, value in obj.items():
                if prop.endswith('_confidence'):
                    continue
                confidence_key = f"{prop}_confidence"
                print("confidence_key",confidence_key)
                confidence = obj.get(confidence_key, 1.0)
                if confidence < 0.8:
                    prompt.append(f"  - {prop}: {value} (Confidence low, please refine or infer)")
                else:
                    prompt.append(f"  - {prop}: {value}")
    
    prompt.append("\nYour task:")
    prompt.append("- Refine the draft description to align with the motion dynamics and physical properties.")
    prompt.append("- Integrate low-confidence physical properties by inferring missing details where necessary.")
    prompt.append("- Ensure the output is descriptive, coherent, and suitable for generating realistic video content.")

    return '\n'.join(prompt)


def save_system_prompt(output_file, prompt):
    with open(output_file, 'w') as file:
        file.write(prompt)
    return file


def create_content(image_folder_path, system_content):
        query = encode_txt(system_content)
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
        return content


def query_gpt(content):
    response = openai.ChatCompletion.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": f"You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        temperature=0.01,
    )
    return response["choices"][0]["message"]["content"]


def generate_txt(data_path:str):
    data_path = data_path
    yaml_file = os.path.join(data_path, 'physics.yaml')
    origin_text_file = os.path.join(data_path, 'description.txt')
    output_file = os.path.join(data_path, 'system_prompt.txt')
    image_path = os.path.join(data_path, 'frames')

    # Load inputs
    with open(origin_text_file, 'r') as file:
        draft_text = file.read()

    physical_properties = load_physical_properties(yaml_file)

    print(f"Draft Text: {draft_text}")
    print(f"Physical Properties: {physical_properties}")

    # Generate system prompt
    system_prompt = create_system_prompt(draft_text, physical_properties)
    save_system_prompt(output_file, system_prompt)
    print("system promt", output_file)
    content = create_content(image_path, output_file)

    gpt_response = query_gpt(content)

    print("GPT-4o Response:")
    print(gpt_response)
    return gpt_response

        
import requests

num_repeat_times = 130000
prefix = "你好，请介绍一下你自己"
repeat_text = "，"

content = prefix + repeat_text * num_repeat_times

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy-key",
}
data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "你是一个 helpful assistant."},
        {"role": "user", "content": content},
    ],
    "temperature": 0.7,
    "max_tokens": 1,
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.text)

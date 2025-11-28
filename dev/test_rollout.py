from time import sleep

from async_grpc.client import AsyncGrpcClient, LLMRequest, RequestLog
from transformers import AutoTokenizer

TOKENIZER_PATH = "你的本地模型路径或HF模型名"

prompts: list[list[int]] = []
with open("dev/data/prompt.txt", mode="r", encoding="utf-8") as file:
    for line in file:
        prompt: list[int] | object = eval(line)
        if not isinstance(prompt, list):
            print(f"unacceptable data format from text file. {prompt}")
        elif any([not isinstance(value, int) for value in prompt]):
            print(f"unacceptable data format from text file. {prompt}")
        else:
            prompts.append(prompt)

print(f"{len(prompts)} prompts was collected.")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

client: AsyncGrpcClient = AsyncGrpcClient(target_url="http://localhost:26001/")

for idx, prompt in enumerate(prompts):
    client.enqueue(
        request=LLMRequest(
            tokens=prompt,
            rmp_tokens=prompt,
            attention_mask=[],
            position_ids=[],
            request_id=idx,
            temperature=0.99,
            top_k=100,
            top_p=0.90,
            max_new_tokens=4096,
            n=8,
        )
    )

responses: list[RequestLog] = []
while len(responses) != len(prompts):
    sleep(1.0)
    responses.extend(client.collect())
    print(f"Collecting Result - {len(responses)}/{len(prompts)}")

with open(file="dev/data/response.txt", mode="w", encoding="utf-8") as file:
    for log in responses:
        req_tokens = log.request.tokens
        resp_tokens = log.response.tokens

        req_text: str = tokenizer.decode(req_tokens, skip_special_tokens=True)
        resp_text: str = tokenizer.decode(resp_tokens, skip_special_tokens=True)

        req_text: str = req_text.replace("\n", "").replace("\r", "")
        resp_text: str = resp_text.replace("\n", "").replace("\r", "")

        file.write(str(log.request.request_id) + "\n")
        file.write(str(req_tokens) + "\n")
        file.write(str(resp_tokens) + "\n")
        file.write(req_text + "\n")
        file.write(resp_text + "\n")
        file.write("--------------------------------------------\n")

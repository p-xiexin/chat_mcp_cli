curl -N -s \
  --request POST \
  --url https://api.siliconflow.cn/v1/chat/completions \
  --header 'Authorization: Bearer sk-fvysgovrkoqorpbqpubqfbvfkwfwqkpthtmvzsvdwfkmwpzz' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "messages": [
        {"role": "system", "content": "你是一名历史学家，请用严谨的学术口吻回答。"},
        {"role":"user","content":"有诺贝尔数学奖吗？"}
    ]
}'
import requests

def query_project(
    base_url: str,
    project_id: str,
    query: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    retrieval_weight: float = 0.5,
):
    url = f"{base_url}/querySimple/{project_id}"
    payload = {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "retrieval_weight": retrieval_weight,
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    try:
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("请求失败：", e)
        print("响应内容：", response.text)
        return None


# 示例调用
if __name__ == "__main__":
    result = query_project(
        base_url="http://192.168.3.14:8900",
        project_id="qBCrXWawc4fe",
        query="如何安装wsl？",
    )
    print(result)

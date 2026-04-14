import requests

client_id = "gp762nuuoqcoxypju8c569th9wz7q5"
token = "ws3rbcrslvgb1vjq7t58sk9l9b9jt6"

r = requests.get(
    "https://api.twitch.tv/helix/users",
    headers={
        "Authorization": f"Bearer {token}",
        "Client-Id": client_id,
    },
    timeout=10,
)

print(r.status_code)
print(r.text)
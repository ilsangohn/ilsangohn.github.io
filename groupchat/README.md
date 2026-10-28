# 단톡방: 나 + Claude + Codex

맥에서 서버를 띄우면 폰이나 다른 컴퓨터의 브라우저로 접속해서
나, Claude Code, Codex 셋이 한 대화방에서 이야기하고 작업할 수 있습니다.

- 메시지를 보내면 Claude가 먼저 답하고, Codex는 Claude의 답까지 읽은 다음 답합니다.
- `@claude ...` 또는 `@codex ...`로 시작하면 그쪽만 답합니다. 화면의 "Claude만"/"Codex만" 버튼도 같은 역할입니다.
- 대화는 작업 폴더의 `.groupchat.jsonl`에 저장되어서 서버를 다시 켜도 이어집니다.

## 준비물 (맥)

- `claude` (Claude Code CLI)와 `codex` (Codex CLI)가 설치되고 로그인된 상태
- Python 3 (맥에 기본 설치되어 있음, 추가 패키지 필요 없음)

## 실행

```bash
cd groupchat
python3 server.py --workdir ~/projects/내-프로젝트
```

서버가 접속 주소를 출력합니다.

```
http://<맥 주소>:8765/?token=xxxxxxxx
```

이 주소로 한 번 접속하면 토큰이 브라우저에 저장되어 다음부터는 `http://<맥 주소>:8765/`만 열어도 됩니다.
토큰을 고정하려면 `CHAT_TOKEN=원하는값 python3 server.py ...`로 실행하세요.

AI가 작업 폴더의 파일을 직접 고치게 하려면 `--allow-edits`를 붙입니다.
붙이지 않으면 읽고 대화만 합니다.

## 폰에서 접속하기

### 집 와이파이 안에서만 쓸 때

맥과 폰이 같은 와이파이에 있으면 맥의 내부 IP로 접속합니다.

```bash
ipconfig getifaddr en0    # 예: 192.168.0.12
```

폰에서 `http://192.168.0.12:8765/?token=...`를 엽니다.

### 밖에서도 쓸 때: Tailscale (추천)

[Tailscale](https://tailscale.com)은 내 기기들끼리만 연결되는 사설 네트워크를 만들어 줍니다.
포트를 인터넷에 열지 않아도 되어서 안전합니다.

1. 맥과 폰에 Tailscale 앱을 설치하고 같은 계정으로 로그인합니다.
2. 맥의 Tailscale 주소를 확인합니다 (메뉴바 아이콘 또는 `tailscale ip -4`, 예: `100.101.102.103`).
3. 폰에서 `http://100.101.102.103:8765/?token=...`를 엽니다.

Tailscale로만 열고 싶다면 `--host 100.101.102.103`으로 실행하세요.
그러면 같은 와이파이의 다른 기기에서는 접속할 수 없습니다.

### 홈 화면에 앱처럼 추가

- 아이폰: Safari에서 열고 공유 버튼에서 "홈 화면에 추가"를 누릅니다.
- 안드로이드: Chrome 메뉴에서 "홈 화면에 추가"를 누릅니다.

## 맥이 잠들지 않게 하기

맥이 잠자기에 들어가면 폰에서 접속할 수 없습니다. 서버를 띄울 때 `caffeinate`로 감싸세요.

```bash
caffeinate -i python3 server.py --workdir ~/projects/내-프로젝트
```

## 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--workdir` | 현재 폴더 | Claude와 Codex가 작업할 폴더 |
| `--port` | 8765 | 포트 |
| `--host` | 0.0.0.0 | 바인딩 주소 |
| `--token` | 자동 생성 | 접속 토큰 (`CHAT_TOKEN` 환경변수로도 지정) |
| `--allow-edits` | 꺼짐 | 파일 수정 허용 (Claude: `acceptEdits`, Codex: `--full-auto`) |
| `--context` | 40 | AI에게 보여줄 최근 메시지 수 |
| `--timeout` | 600 | AI 한 번 응답 제한 시간(초) |

## 알아둘 점

- AI는 매번 대화 기록을 새로 읽고 답합니다. 대화가 길어지면 최근 `--context`개만 봅니다.
- 한 번에 한 라운드씩 처리합니다. AI가 답하는 중에 보낸 메시지는 앞 라운드가 끝난 뒤 처리됩니다.
- 토큰을 아는 사람은 누구나 AI에게 맥에서 작업을 시킬 수 있습니다. 주소를 남에게 공유하지 마세요.

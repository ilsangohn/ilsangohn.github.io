#!/usr/bin/env python3
"""나 + Claude + Codex 단톡방 서버.

맥에서 실행하면 브라우저(폰 포함)로 접속해 세 명이 한 대화방에서 이야기할 수 있다.
외부 패키지 없이 파이썬 표준 라이브러리만 사용한다.

    python3 server.py --workdir ~/projects/my-app
"""
import argparse
import hmac
import json
import os
import pathlib
import secrets
import subprocess
import tempfile
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = pathlib.Path(__file__).resolve().parent
AGENTS = ("claude", "codex")
NAMES = {"me": "나", "claude": "Claude", "codex": "Codex"}

messages = []          # {"id", "who", "text", "ts"}
status = {"busy": None}  # 지금 답변 중인 에이전트
lock = threading.Lock()
round_lock = threading.Lock()  # 한 번에 한 라운드씩 처리
args = None


def add_message(who, text):
    with lock:
        msg = {"id": len(messages), "who": who, "text": text, "ts": time.time()}
        messages.append(msg)
        with args.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    return msg


def load_log():
    if not args.log.exists():
        return
    for line in args.log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            msg = json.loads(line)
            msg["id"] = len(messages)
            messages.append(msg)


def transcript():
    with lock:
        recent = messages[-args.context:]
    return "\n\n".join(f"[{NAMES[m['who']]}]\n{m['text']}" for m in recent)


def build_prompt(agent):
    other = "Codex" if agent == "claude" else "Claude"
    return (
        f"너는 {NAMES[agent]}이고, 지금 사용자('나'), {other}와 함께 셋이 쓰는 그룹 채팅방에 있어.\n"
        f"작업 폴더는 {args.workdir} 야.\n"
        f"아래 대화를 읽고 {NAMES[agent]}로서 다음 발언 하나만 해줘. "
        f"{other}의 의견에 동의하거나 반박하거나 보완해도 좋아. "
        f"이름표([{NAMES[agent]}])는 붙이지 말고 내용만 써.\n\n"
        f"--- 대화 기록 (최근 {args.context}개) ---\n{transcript()}"
    )


def ask(agent, prompt):
    try:
        if agent == "claude":
            cmd = ["claude", "-p", prompt]
            if args.allow_edits:
                cmd += ["--permission-mode", "acceptEdits"]
            r = subprocess.run(cmd, cwd=args.workdir, capture_output=True,
                               text=True, timeout=args.timeout)
            out = r.stdout.strip()
        else:
            with tempfile.TemporaryDirectory() as d:
                last = pathlib.Path(d) / "last.txt"
                cmd = ["codex", "exec", "--skip-git-repo-check",
                       "--output-last-message", str(last)]
                if args.allow_edits:
                    cmd.append("--full-auto")
                cmd.append(prompt)
                r = subprocess.run(cmd, cwd=args.workdir, capture_output=True,
                                   text=True, timeout=args.timeout)
                out = last.read_text().strip() if last.exists() else ""
        if not out:
            out = f"(응답 없음, 종료 코드 {r.returncode})\n{r.stderr.strip()[-1500:]}"
        return out
    except FileNotFoundError:
        return f"({agent} 명령을 찾을 수 없어요. 설치와 PATH를 확인하세요.)"
    except subprocess.TimeoutExpired:
        return f"({args.timeout}초 안에 답하지 못했어요.)"


def run_round(targets):
    with round_lock:
        for agent in targets:
            status["busy"] = agent
            try:
                add_message(agent, ask(agent, build_prompt(agent)))
            finally:
                status["busy"] = None


def pick_targets(text):
    t = text.lstrip().lower()
    chosen = [a for a in AGENTS if t.startswith(f"@{a}")]
    return chosen or list(AGENTS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def authorized(self, query):
        token = query.get("token", [None])[0]
        if token is None:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            token = cookie["chat_token"].value if "chat_token" in cookie else ""
        return hmac.compare_digest(token or "", args.token)

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self.authorized(query):
            self.send_response(401)
            self.end_headers()
            self.wfile.write("토큰이 필요해요. 서버가 출력한 주소로 접속하세요.".encode())
            return
        if url.path == "/":
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            if "token" in query:
                # 한 번 토큰으로 들어오면 쿠키로 기억 (1년)
                self.send_header("Set-Cookie", f"chat_token={args.token}; Max-Age=31536000; "
                                 "HttpOnly; SameSite=Strict; Path=/")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/messages":
            since = int(query.get("since", ["0"])[0])
            with lock:
                new = messages[since:]
            self.send_json({"messages": new, "busy": status["busy"],
                            "workdir": str(args.workdir)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        if not self.authorized(parse_qs(url.query)):
            self.send_json({"error": "unauthorized"}, 401)
            return
        if url.path != "/send":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        text = json.loads(self.rfile.read(length) or b"{}").get("text", "").strip()
        if not text:
            self.send_json({"error": "empty"}, 400)
            return
        add_message("me", text)
        threading.Thread(target=run_round, args=(pick_targets(text),), daemon=True).start()
        self.send_json({"ok": True})


def main():
    global args
    p = argparse.ArgumentParser(description="나 + Claude + Codex 단톡방")
    p.add_argument("--host", default="0.0.0.0",
                   help="바인딩 주소 (Tailscale IP만 열려면 100.x.x.x)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path.cwd(),
                   help="Claude와 Codex가 작업할 폴더")
    p.add_argument("--token", default=os.environ.get("CHAT_TOKEN"),
                   help="접속 토큰 (기본: CHAT_TOKEN 환경변수, 없으면 자동 생성)")
    p.add_argument("--context", type=int, default=40, help="AI에게 보여줄 최근 메시지 수")
    p.add_argument("--timeout", type=int, default=600, help="AI 한 번 응답 제한 시간(초)")
    p.add_argument("--allow-edits", action="store_true",
                   help="AI가 작업 폴더의 파일을 수정하도록 허용")
    args = p.parse_args()
    args.workdir = args.workdir.expanduser().resolve()
    args.log = args.workdir / ".groupchat.jsonl"
    if not args.token:
        args.token = secrets.token_urlsafe(16)

    load_log()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"작업 폴더: {args.workdir}")
    print(f"대화 기록: {args.log}")
    print(f"파일 수정 허용: {'예' if args.allow_edits else '아니오'}")
    print(f"\n접속 주소 (이 주소를 폰에서 여세요):")
    print(f"  http://<맥 주소>:{args.port}/?token={args.token}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

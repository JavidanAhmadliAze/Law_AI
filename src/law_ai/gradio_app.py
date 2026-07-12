"""Gradio chat UI, mounted on the FastAPI app at /ui.

Layout: login screen → left sidebar with the user's chats (select / new /
delete) → right pane with the conversation thread and question box.

The UI talks to the app's own REST API in-process (ASGITransport — no
network hop), so auth, ownership checks and the RAG pipeline are exercised
exactly as any other API client would.
"""

from typing import Any

import gradio as gr
import httpx
from fastapi import FastAPI


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://law-ai.internal"
    )


def build_gradio_ui(app: FastAPI) -> gr.Blocks:
    async def do_login(username: str, password: str) -> tuple[Any, ...]:
        async with _client(app) as client:
            response = await client.post(
                "/auth/login", json={"username": username, "password": password}
            )
        if response.status_code != 200:
            return (
                None,
                gr.update(visible=True),
                gr.update(visible=False),
                (f"❌ {response.json().get('detail', 'Login failed')}"),
                gr.update(choices=[]),
            )
        token = response.json()["access_token"]
        chats = await fetch_chats(token)
        return (
            token,
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            gr.update(choices=chats, value=None),
        )

    async def do_register(username: str, password: str) -> str:
        async with _client(app) as client:
            response = await client.post(
                "/auth/register", json={"username": username, "password": password}
            )
        if response.status_code == 201:
            return "✅ Registered — now log in."
        return f"❌ {response.json().get('detail', 'Registration failed')}"

    async def fetch_chats(token: str) -> list[tuple[str, str]]:
        async with _client(app) as client:
            response = await client.get("/chats", headers={"authorization": f"Bearer {token}"})
        if response.status_code != 200:
            return []
        return [(c["title"], c["id"]) for c in response.json()]

    async def new_chat(token: str) -> tuple[Any, ...]:
        async with _client(app) as client:
            response = await client.post(
                "/chats", json={}, headers={"authorization": f"Bearer {token}"}
            )
        chat_id = response.json()["id"]
        chats = await fetch_chats(token)
        return gr.update(choices=chats, value=chat_id), []

    async def select_chat(token: str, chat_id: str | None) -> list[dict[str, str]]:
        if not chat_id:
            return []
        async with _client(app) as client:
            response = await client.get(
                f"/chats/{chat_id}/messages",
                headers={"authorization": f"Bearer {token}"},
            )
        if response.status_code != 200:
            return []
        return [{"role": m["role"], "content": m["content"]} for m in response.json()]

    async def delete_chat(token: str, chat_id: str | None) -> tuple[Any, ...]:
        if chat_id:
            async with _client(app) as client:
                await client.delete(
                    f"/chats/{chat_id}", headers={"authorization": f"Bearer {token}"}
                )
        chats = await fetch_chats(token)
        return gr.update(choices=chats, value=None), []

    async def send(
        token: str, chat_id: str | None, question: str, thread: list[dict[str, str]]
    ) -> tuple[Any, ...]:
        if not question.strip():
            return thread, "", gr.update()
        if not chat_id:  # auto-create a chat for the first question
            async with _client(app) as client:
                response = await client.post(
                    "/chats", json={}, headers={"authorization": f"Bearer {token}"}
                )
            chat_id = response.json()["id"]

        thread = [*thread, {"role": "user", "content": question}]
        async with _client(app) as client:
            response = await client.post(
                f"/chats/{chat_id}/ask",
                json={"question": question},
                headers={"authorization": f"Bearer {token}"},
                timeout=300.0,
            )
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            if data.get("citations"):
                answer += "\n\n**Sources:**\n" + "\n".join(
                    f"- *{c['article']}*: „{c['quote']}”" for c in data["citations"]
                )
        else:
            answer = f"⚠️ {response.json().get('detail', 'Something went wrong')}"
        thread = [*thread, {"role": "assistant", "content": answer}]
        chats = await fetch_chats(token)
        return thread, "", gr.update(choices=chats, value=chat_id)

    with gr.Blocks(title="Law-AI — Polish Law Assistant") as ui:
        token_state = gr.State(None)

        # ---------------- login screen ----------------
        with gr.Column(visible=True) as login_view:
            gr.Markdown("# ⚖️ Law-AI\nAsk questions about Polish law.")
            username_in = gr.Textbox(label="Username")
            password_in = gr.Textbox(label="Password", type="password")
            with gr.Row():
                login_btn = gr.Button("Log in", variant="primary")
                register_btn = gr.Button("Register")
            login_status = gr.Markdown("")

        # ---------------- chat screen ----------------
        with gr.Row(visible=False) as chat_view:
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Chats")
                chat_list = gr.Radio(choices=[], label="", interactive=True)
                new_btn = gr.Button("＋ New chat")
                delete_btn = gr.Button("🗑 Delete chat", variant="stop")
            with gr.Column(scale=4):
                thread = gr.Chatbot(label="", height=520)  # gradio 6: messages format is default
                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="Ask about Polish law…",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        # ---------------- wiring ----------------
        login_btn.click(
            do_login,
            inputs=[username_in, password_in],
            outputs=[token_state, login_view, chat_view, login_status, chat_list],
        )
        register_btn.click(do_register, inputs=[username_in, password_in], outputs=[login_status])
        new_btn.click(new_chat, inputs=[token_state], outputs=[chat_list, thread])
        chat_list.change(select_chat, inputs=[token_state, chat_list], outputs=[thread])
        delete_btn.click(delete_chat, inputs=[token_state, chat_list], outputs=[chat_list, thread])
        send_btn.click(
            send,
            inputs=[token_state, chat_list, question_box, thread],
            outputs=[thread, question_box, chat_list],
        )
        question_box.submit(
            send,
            inputs=[token_state, chat_list, question_box, thread],
            outputs=[thread, question_box, chat_list],
        )

    return ui


def mount_gradio(app: FastAPI) -> None:
    gr.mount_gradio_app(app, build_gradio_ui(app), path="/ui")

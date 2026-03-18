"""Webhook route registration for n8n / WhatsApp integration."""

from __future__ import annotations

from typing import Any, Dict


def register_routes(app) -> None:
    """Register webhook and health routes on a Flask app instance."""

    @app.post("/webhook/whatsapp")
    def whatsapp_webhook():
        from flask import current_app, jsonify, request

        payload = request.get_json(silent=True) or {}
        user_phone_number = _extract_field(payload, "user_phone_number")
        message_body = _extract_field(payload, "message_body")

        if not message_body:
            return jsonify({"error": "message_body is required"}), 400

        engine = current_app.extensions["sfim_engine"]
        result = engine.answer(message_body, user_phone_number=user_phone_number)

        return jsonify(
            {
                "status": "ok",
                "user_phone_number": user_phone_number,
                "message_body": message_body,
                "reply_message": result.answer,
                "xml_matches": result.xml_hits,
                "pdf_matches": result.pdf_hits,
                "self_evaluation_summary": getattr(result, "self_evaluation_summary", ""),
                "self_evaluation": getattr(result, "self_evaluation", {}),
                "refinement_memory": getattr(result, "refinement_memory", []),
            }
        )


def _extract_field(payload: Dict[str, Any], key: str) -> str:
    if key in payload and payload.get(key) is not None:
        return str(payload.get(key)).strip()

    body = payload.get("body")
    if isinstance(body, dict) and key in body and body.get(key) is not None:
        return str(body.get(key)).strip()

    data = payload.get("data")
    if isinstance(data, dict) and key in data and data.get(key) is not None:
        return str(data.get(key)).strip()

    return ""

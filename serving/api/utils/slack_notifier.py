import httpx
from typing import List, Dict, Any
from config.settings import SLACK_WEBHOOK_URL, SLACK_ALERT_ENABLED


async def send_slack_alert(
    status: str,
    alerts: List[Dict[str, Any]],
    session_info: Dict[str, Any],
    status_change_message: str = None
):
    """
    Slack 웹훅으로 알림 전송

    Args:
        status: 시스템 상태 (healthy, warning, critical)
        alerts: 알림 목록
        session_info: 세션 정보
        status_change_message: 상태 변화 메시지 (선택사항)
    """
    if not SLACK_ALERT_ENABLED or not SLACK_WEBHOOK_URL:
        return

    # 상태 이모지
    status_emoji = {
        "healthy": "✅",
        "warning": "⚠️",
        "critical": "🚨"
    }

    # 메시지 헤더
    if status_change_message:
        header = f"{status_emoji.get(status, '🔔')} {status_change_message}"
    else:
        header = f"{status_emoji.get(status, '🔔')} PCB 검사 시스템 알림"

    # 세션 정보 포맷
    session_id = session_info.get("id", "N/A")
    if session_id == "N/A":
        session_display = "전체 데이터"
    elif session_id is None:
        session_display = "세션 없음"
    else:
        is_active = session_info.get("is_active", False)
        session_display = f"세션 {session_id}" + (" (활성)" if is_active else "")

    # 알림 메시지 구성
    alert_text = ""
    if alerts:
        critical_alerts = [a for a in alerts if a["level"] == "critical"]
        warning_alerts = [a for a in alerts if a["level"] == "warning"]

        if critical_alerts:
            alert_text += "🔴 *Critical 알림:*\n"
            for alert in critical_alerts:
                alert_text += f"• {alert['message']}\n"
                if "action" in alert:
                    alert_text += f"  → {alert['action']}\n"
                alert_text += "\n"

        if warning_alerts:
            alert_text += "🟡 *Warning 알림:*\n"
            for alert in warning_alerts:
                alert_text += f"• {alert['message']}\n"
                if "action" in alert:
                    alert_text += f"  → {alert['action']}\n"
                alert_text += "\n"

    if not alert_text:
        alert_text = "알림 없음"

    # Slack 메시지 페이로드
    message = {
        "text": header,
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": header
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*상태:* {status}"},
                    {"type": "mrkdwn", "text": f"*세션:* {session_display}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert_text
                }
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SLACK_WEBHOOK_URL, json=message, timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            print(f"Slack 알림 전송 실패: {e}")

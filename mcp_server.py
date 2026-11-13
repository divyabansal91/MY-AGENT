"""MCP server: exposes the business tools to the AI agent over stdio.

Run on its own for debugging:  python mcp_server.py
The agent (app.py) starts it as a subprocess and talks to it via MCP.
"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

import rag

BASE = Path(__file__).parent
CATALOG = json.loads((BASE / "data" / "services.json").read_text(encoding="utf-8"))
BOOKINGS_FILE = BASE / "data" / "bookings.json"
IST = timezone(timedelta(hours=5, minutes=30))

mcp = FastMCP("servicemate-tools")


def _today() -> date:
    return datetime.now(IST).date()


def _load_bookings():
    try:
        return json.loads(BOOKINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_bookings(rows):
    BOOKINGS_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _find_service(text: str):
    t = text.strip().lower()
    for s in CATALOG["services"]:
        if t in (s["id"], s["name"].lower()) or t in s["name"].lower() or s["id"] in t:
            return s
    return None


def _check_date(d: str):
    """Return (date, error). Bookings: Mon-Sat, today or later, within 30 days."""
    try:
        day = datetime.strptime(d.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None, "Date must be in YYYY-MM-DD format."
    if day < _today():
        return None, "That date is in the past."
    if day > _today() + timedelta(days=30):
        return None, "We only take bookings up to 30 days ahead."
    if day.weekday() == 6:
        return None, "We are closed on Sundays. Please pick Monday to Saturday."
    return day, None


def _free_slots(service_id: str, day: date):
    taken = {
        b["time"] for b in _load_bookings()
        if b["service_id"] == service_id and b["date"] == day.isoformat()
    }
    return [s for s in CATALOG["slots"] if s not in taken]


@mcp.tool()
def search_knowledge_base(
    query: Annotated[str, Field(description="What to look up, e.g. 'cancellation policy' or 'AC repair price'")],
) -> str:
    """Search the company knowledge base (services, prices, policies, FAQ) using RAG.
    Always use this before answering questions about prices, policies, warranty or coverage."""
    hits = rag.search(query, top_k=3)
    if not hits:
        return json.dumps({"results": [], "note": "Nothing relevant found."})
    return json.dumps({"results": hits}, ensure_ascii=False)


@mcp.tool()
def list_services() -> str:
    """List all services we offer with their price and duration."""
    return json.dumps(CATALOG["services"], ensure_ascii=False)


@mcp.tool()
def check_availability(
    service: Annotated[str, Field(description="Service name or id, e.g. 'plumbing'")],
    date_str: Annotated[str, Field(description="Date in YYYY-MM-DD format")],
) -> str:
    """Check which time slots are free for a service on a given date."""
    s = _find_service(service)
    if not s:
        return json.dumps({"error": "Unknown service.", "services": [x["name"] for x in CATALOG["services"]]})
    day, err = _check_date(date_str)
    if err:
        return json.dumps({"error": err})
    return json.dumps({
        "service": s["name"],
        "date": day.isoformat(),
        "weekday": day.strftime("%A"),
        "free_slots": _free_slots(s["id"], day),
    })


@mcp.tool()
def create_booking(
    customer_name: Annotated[str, Field(description="Customer full name")],
    phone: Annotated[str, Field(description="10-digit mobile number")],
    service: Annotated[str, Field(description="Service name or id")],
    date_str: Annotated[str, Field(description="Date in YYYY-MM-DD format")],
    time_slot: Annotated[str, Field(description="Slot in HH:MM, must be one returned by check_availability")],
) -> str:
    """Create a booking. Only call after the customer has confirmed all details."""
    s = _find_service(service)
    if not s:
        return json.dumps({"error": "Unknown service."})
    day, err = _check_date(date_str)
    if err:
        return json.dumps({"error": err})
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) != 10:
        return json.dumps({"error": "Phone number must be 10 digits."})
    if not customer_name.strip():
        return json.dumps({"error": "Customer name is required."})
    if time_slot not in _free_slots(s["id"], day):
        return json.dumps({"error": "Slot not available.", "free_slots": _free_slots(s["id"], day)})
    rows = _load_bookings()
    booking = {
        "booking_id": f"SM{len(rows) + 1001}",
        "customer_name": customer_name.strip(),
        "phone": digits,
        "service_id": s["id"],
        "service": s["name"],
        "date": day.isoformat(),
        "time": time_slot,
        "status": "confirmed",
    }
    rows.append(booking)
    _save_bookings(rows)
    return json.dumps(booking, ensure_ascii=False)


@mcp.tool()
def get_booking(
    booking_id: Annotated[str, Field(description="Booking ID like SM1001")],
) -> str:
    """Look up an existing booking by its ID."""
    for b in _load_bookings():
        if b["booking_id"].lower() == booking_id.strip().lower():
            return json.dumps(b, ensure_ascii=False)
    return json.dumps({"error": "Booking not found."})


if __name__ == "__main__":
    mcp.run(transport="stdio")
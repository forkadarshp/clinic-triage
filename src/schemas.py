"""Pydantic schemas for the 3 triage tools - strict validation."""

from enum import Enum
from typing import Literal, Union
from pydantic import BaseModel, Field, field_validator


class ToolName(str, Enum):
    """Valid tool names for triage routing."""
    EMERGENCY = "trigger_emergency_response"
    URGENT = "schedule_urgent_consult"
    ROUTINE = "routine_care_referral"


# =============================================================================
# Tool 1: Emergency Response
# =============================================================================
class EmergencyResponseArgs(BaseModel):
    """Arguments for trigger_emergency_response."""
    location: str = Field(..., min_length=1, description="Patient location")
    severity: Literal["CRITICAL"] = Field(
        default="CRITICAL",
        description="Severity level (always CRITICAL for emergencies)"
    )


class EmergencyResponse(BaseModel):
    """Tool call for life-threatening cases (heart attack, stroke, trauma)."""
    tool: Literal["trigger_emergency_response"] = Field(
        default="trigger_emergency_response"
    )
    arguments: EmergencyResponseArgs


# =============================================================================
# Tool 2: Urgent Consult
# =============================================================================
class UrgentConsultArgs(BaseModel):
    """Arguments for schedule_urgent_consult."""
    department: str = Field(..., min_length=1, description="Target department")
    symptoms: list[str] = Field(
        ...,
        min_length=1,
        description="List of presenting symptoms"
    )

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v: list[str]) -> list[str]:
        """Ensure non-empty symptom strings."""
        return [s.strip() for s in v if s.strip()]


class UrgentConsult(BaseModel):
    """Tool call for serious but non-fatal cases (high fever, fractures)."""
    tool: Literal["schedule_urgent_consult"] = Field(
        default="schedule_urgent_consult"
    )
    arguments: UrgentConsultArgs


# =============================================================================
# Tool 3: Routine Care
# =============================================================================
class RoutineCareArgs(BaseModel):
    """Arguments for routine_care_referral."""
    type: str = Field(..., min_length=1, description="Type of care needed")
    specialty: str = Field(..., min_length=1, description="Medical specialty")


class RoutineCare(BaseModel):
    """Tool call for chronic conditions, refills, checkups."""
    tool: Literal["routine_care_referral"] = Field(
        default="routine_care_referral"
    )
    arguments: RoutineCareArgs


# =============================================================================
# Union Type for Parsing
# =============================================================================
TriageOutput = Union[EmergencyResponse, UrgentConsult, RoutineCare]


def parse_triage_output(data: dict) -> TriageOutput:
    """
    Parse and validate a triage output dictionary.
    
    Args:
        data: Dictionary with 'tool' and 'arguments' keys
        
    Returns:
        Validated TriageOutput (one of the 3 tool types)
        
    Raises:
        ValueError: If tool name is invalid
        ValidationError: If arguments don't match schema
    """
    tool_name = data.get("tool", "")
    
    if tool_name == ToolName.EMERGENCY.value:
        return EmergencyResponse(**data)
    elif tool_name == ToolName.URGENT.value:
        return UrgentConsult(**data)
    elif tool_name == ToolName.ROUTINE.value:
        return RoutineCare(**data)
    else:
        raise ValueError(f"Invalid tool name: '{tool_name}'. Must be one of: {[t.value for t in ToolName]}")


def get_mock_response(output: TriageOutput) -> str:
    """
    Generate a mock system response for a validated triage output.
    
    Args:
        output: Validated TriageOutput
        
    Returns:
        Mock confirmation string
    """
    if isinstance(output, EmergencyResponse):
        return f"[System] 🚑 Ambulance dispatched to {output.arguments.location}. ETA: 5 minutes."
    elif isinstance(output, UrgentConsult):
        symptoms_str = ", ".join(output.arguments.symptoms)
        return f"[System] 📋 Urgent consult scheduled with {output.arguments.department}. Symptoms: {symptoms_str}"
    elif isinstance(output, RoutineCare):
        return f"[System] 📅 Routine {output.arguments.type} referral created for {output.arguments.specialty}."
    else:
        return "[System] ❓ Unknown action."

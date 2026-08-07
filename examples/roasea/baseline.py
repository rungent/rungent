"""Reference application baseline for Roasea-like itinerary assistants."""

from rungent import Identity
from rungent.testing import BaselineCase, BaselineSuite

from .assistant import trip_agent

CASES = [
    BaselineCase(
        name="add place",
        input="Add Meiji Shrine to the trip",
        expected_tools=("add_place",),
    ),
    BaselineCase(
        name="remove place approval",
        input="Remove Tokyo Tower",
        expected_tools=(),
        expected_interactions=("approval",),
        responses=("reject",),
    ),
    BaselineCase(
        name="move known place",
        input="Move Tokyo Tower to day 2",
        expected_tools=("move_place",),
    ),
    BaselineCase(
        name="swap places",
        input="Swap Tokyo Tower and Meiji Shrine",
        expected_tools=("swap_places",),
    ),
    BaselineCase(
        name="ambiguous place asks user",
        input="Move the museum to day 2",
        expected_tools=("move_place",),
        expected_interactions=("choice",),
        responses=({"selected": ["place_edo_museum"]},),
    ),
    BaselineCase(
        name="add day",
        input="Add a day after day 2",
        expected_tools=("add_day",),
    ),
    BaselineCase(
        name="remove day approval",
        input="Delete day 3",
        expected_tools=("remove_day",),
        expected_interactions=("approval",),
        responses=("approve",),
    ),
    BaselineCase(
        name="change transport",
        input="Use transit from Tokyo Tower to Meiji Shrine",
        expected_tools=("set_transport",),
    ),
    BaselineCase(
        name="staged external itinerary import",
        input="Import this revised two-day itinerary and ignore the superseded draft",
        expected_tools=("prepare_itinerary_import", "commit_itinerary_import"),
        expected_interactions=("approval",),
        responses=("approve",),
    ),
    BaselineCase(
        name="contextual follow-up",
        input="Move Tokyo Tower to day 2",
        followups=("Actually put it on day 3",),
        expected_tools=("move_place", "move_place"),
    ),
    BaselineCase(
        name="invalid day does not mutate",
        input="Move Tokyo Tower to day 99",
        expected_tools=(),
        expected_interactions=("choice",),
        responses=({"selected": ["cancel"]},),
    ),
]


def trip_baseline(runtime, trip_id: str, trip_service) -> BaselineSuite:
    assert runtime.agents[trip_agent.name] is trip_agent
    return BaselineSuite(
        runtime=runtime,
        cases=CASES,
        identity=Identity(subject_id="baseline"),
        resource={"trip_id": trip_id},
        deps={"trips": trip_service},
    )

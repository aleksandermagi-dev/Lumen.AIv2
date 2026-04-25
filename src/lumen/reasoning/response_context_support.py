from __future__ import annotations


class ResponseContextSupport:
    """Support-only response context attachment; does not choose route or lane."""

    @staticmethod
    def attach_profile_context(
        *,
        response: dict[str, object],
        interaction_profile,
        profile_advice,
        client_surface: str,
    ) -> None:
        response["interaction_profile"] = interaction_profile.to_dict()
        response["client_surface"] = client_surface
        if profile_advice:
            response["profile_advice"] = dict(profile_advice)

    @staticmethod
    def attach_wake_metadata(
        *,
        response: dict[str, object],
        wake_interaction: dict[str, object] | None,
    ) -> None:
        if not wake_interaction:
            return
        response["wake_interaction"] = {
            "wake_phrase": wake_interaction.get("wake_phrase"),
            "classification": wake_interaction.get("classification"),
            "stripped_prompt": wake_interaction.get("stripped_prompt"),
        }

    @staticmethod
    def attach_wake_resolution(
        *,
        response: dict[str, object],
        original_prompt: str,
        effective_prompt: str,
        wake_interaction: dict[str, object] | None,
    ) -> None:
        if not wake_interaction:
            return
        if str(wake_interaction.get("classification") or "") != "greeting_plus_request":
            return
        if not effective_prompt or effective_prompt == original_prompt:
            return
        response["resolved_prompt"] = effective_prompt
        response["resolution_strategy"] = "wake_phrase_strip"
        response["resolution_reason"] = (
            "Leading wake phrase was removed before routing so engine selection could evaluate the actual request."
        )

    @staticmethod
    def attach_route_metadata(
        *,
        response: dict[str, object],
        route,
        route_status: str | None,
    ) -> None:
        response["route"] = route.to_metadata().to_dict()
        if route_status:
            response["route"]["status"] = route_status

    @staticmethod
    def attach_response_behavior_posture(
        *,
        response: dict[str, object],
        route_mode: str,
        response_strategy_layer,
        low_confidence_recovery=None,
        srd_diagnostic=None,
        state_control=None,
    ) -> None:
        # This may describe strategy posture, but it must not change the selected mode/kind.
        posture = response_strategy_layer.select(
            mode=str(response.get("mode") or "").strip(),
            route_mode=str(route_mode or "").strip(),
            low_confidence_recovery=low_confidence_recovery,
            srd_diagnostic=srd_diagnostic,
            state_control=state_control,
            response_payload=response,
        )
        response["response_behavior_posture"] = posture.to_dict()

from __future__ import annotations


class ResponsePackagingSupport:
    """Turn-level reply packaging and observability helpers.

    This layer owns interactive response packaging only. It may add metadata and
    presentation context for the current assistant turn, but it must never alter
    route authority or take over durable report/artifact formatting.
    """

    @staticmethod
    def attach_execution_and_packaging(
        *,
        response: dict[str, object],
        pipeline_trace,
        mode: str,
        kind: str,
        route,
        interaction_profile,
        reasoning_pipeline,
    ) -> None:
        execution_stage = reasoning_pipeline.package_execution_stage(
            mode=mode,
            kind=kind,
            response_payload=response,
        )
        response["pipeline_execution"] = execution_stage.to_dict()
        reasoning_pipeline.record_execution_package(pipeline_trace, execution_stage)
        response_package = reasoning_pipeline.package_response(
            mode=mode,
            kind=kind,
            route=route,
            response_payload=response,
            interaction_profile=interaction_profile,
        )
        response["pipeline_packaging"] = response_package.to_dict()
        reasoning_pipeline.record_response_package(pipeline_trace, response_package)

    @staticmethod
    def attach_persistence_observation(
        *,
        response: dict[str, object],
        pipeline_trace,
        session_id: str,
        prompt: str,
        front_half,
        route,
        clarification_decision,
        validation_context,
        reasoning_frame_assembly,
        reasoning_pipeline,
    ) -> None:
        observation = reasoning_pipeline.package_persistence_observation(
            session_id=session_id,
            prompt=prompt,
            front_half=front_half,
            route=route,
            clarification_decision=clarification_decision,
            validation_context=validation_context,
            reasoning_frame_assembly=reasoning_frame_assembly,
            execution_stage=pipeline_trace.execution_package,
            response_packaging=pipeline_trace.response_package,
        )
        response["pipeline_observability"] = observation.to_dict()
        reasoning_pipeline.record_persistence_observation(pipeline_trace, observation)

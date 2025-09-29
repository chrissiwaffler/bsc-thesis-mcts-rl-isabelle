from typing import Any

import lilypad
from mirascope import llm
from pydantic import BaseModel, Field

from beamsearch.llm.prompts import suggest_next_tactic
from beamsearch.utils.parser import is_valid_isabelle_command
from beamsearch.utils.proof import (
    extract_statement_without_proof,
    would_be_duplicate_statement,
)

from ..core import (
    BeamSearchConfig,
    ProofGraph,
    ProofState,
    QIsabelleServerError,
    QIsabelleSession,
)
from ..tools import (
    IsabelleErrorResult,
    IsabelleExecuteCommand,
    IsabelleSuccessResult,
    SledgehammerManager,
)
from ..utils import (
    ProofTreeVisualizer,
    get_proof_path_to_current,
    parse_isabelle_response,
)
from .constants import MODEL, PROVIDER
from .critics import evaluate_command_in_context
from .prompts import reflect_on_failures


class IsabelleProofAgent(BaseModel):
    """Agent for automated Isabelle theorem proving with beam search"""

    # agent state
    graph: ProofGraph
    session: QIsabelleSession
    viz: ProofTreeVisualizer | None = None
    theorem: str
    config: BeamSearchConfig = Field(default_factory=BeamSearchConfig)
    sledgehammer_manager: SledgehammerManager | None = None

    # current search state
    current_depth: int = 0
    exploration_history: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sledgehammer_manager = SledgehammerManager(self.session, max_workers=4)

    def _update_viz(self, title: str):
        """safely update visualization if viz is not None"""
        if self.viz:
            self.viz.update(self.graph, title)

    # TODO: remove this
    # def _execute_isabelle_command(
    #     self, current_state: str, command: str, new_state: str
    # ) -> IsabelleResult:
    #     """Execute an Isabelle command and return result"""
    #     tool = IsabelleExecuteCommand(
    #         current_state=current_state,
    #         command=command,
    #         new_state=new_state,
    #         session=self.session,
    #     )
    #     return tool.call()

    # def _execute_command_with_logging(
    #     self, parent_state: ProofState, command: str, child: ProofState
    # ) -> IsabelleResult:
    #     """Execute an Isabelle command with logging"""
    #
    #     return result

    def _process_command(
        self,
        parent_state: ProofState,
        command: str,
        temperature: float,
        all_candidates: list[ProofState],
    ) -> ProofState | None:
        """Process a single candidate command and return the child state if successful

        Returns:
            The child ProofState if successful, None if failed
        """

        child = self.graph.add_state(
            command=command,
            parent=parent_state.state_name,
            temperature=temperature,
        )

        # execute command with logging via lilypad
        with lilypad.span("Execute Command") as span:
            span.metadata(
                {
                    "command": command,
                    "parent_state": parent_state.state_name,
                    "temperature": child.temperature,
                    "source": "llm_generation",
                }
            )

            result = IsabelleExecuteCommand(
                current_state=parent_state.state_name,
                command=command,
                new_state=child.state_name,
                session=self.session,
            ).call()
            print(f"Debug: Raw result = {result}")

            self._update_viz(f"Trying: {command[:50]}...")

            if isinstance(result, IsabelleErrorResult):
                span.error(f"Command failed: {result.error[:200]}")

                return self._handle_failed_command(
                    parent_state,
                    child,
                    command,
                    result.error,
                    all_candidates,
                    temperature,
                )
            else:
                return self._handle_successful_command(
                    parent_state, child, command, result, all_candidates
                )

    # TODO: check if all_candidates is somewhere really used ????

    def _handle_failed_command(
        self,
        parent_state: ProofState,
        child: ProofState,
        command: str,
        error_msg: str,
        all_candidates: list[ProofState],
        temperature: float,
    ):
        self.graph.add_failed_tactic(parent_state.state_name, command, error_msg)
        parent_state.failure_error_messages.append(error_msg)

        print(f"FAILED: {command} with error: {error_msg}")

        # TODO: check this "hack"
        # try fallback strategy if in state mode
        if parent_state.mode == "state":
            fallback_result = self._try_statement_without_proof_fallback(
                parent_state, command, temperature, all_candidates
            )
            if fallback_result:
                # TODO: could: remove the original failed child here
                return fallback_result

        self._remove_failed_child(parent_state, child)
        self._update_viz(f"Failed: {command[:40]}")
        return None

    def _try_statement_without_proof_fallback(
        self,
        parent_state: ProofState,
        original_command: str,
        temperature: float,
        all_candidates: list[ProofState],
    ) -> ProofState | None:
        """try executing a statement without its proof"""
        statement_without_proof = extract_statement_without_proof(original_command)

        if not statement_without_proof:
            return None

        # check if we can and should try this out
        if statement_without_proof in [
            t[0] for t in parent_state.failed_tactics
        ] or would_be_duplicate_statement(
            statement_without_proof, parent_state, self.graph
        ):
            return None

        print(f"FALLBACK: trying statement without proof: {statement_without_proof}")

        # create and execute fallback
        return self._process_command(
            parent_state, statement_without_proof, temperature, all_candidates
        )

    def _handle_successful_command(
        self,
        parent_state: ProofState,
        child: ProofState,
        command: str,
        result: IsabelleSuccessResult,
        all_candidates: list[ProofState],
    ) -> ProofState:
        """handle a successful command execution"""
        self.graph.update_result(child.state_name, result.is_done, result.result)

        self._check_auto_done(result, child, all_candidates)

        child.probability = evaluate_command_in_context(
            command=command,
            proof_path=get_proof_path_to_current(self.graph, parent_state.state_name),
            current_goals=parent_state.result,
            theorem=self.theorem,
            result_after_command=result.result,
        )

        all_candidates.append(child)

        print(f"SUCCESSFULLY applying command {command}")
        self._update_viz(f"SUCCESSFULLY applying command {command}")

        return child

    def _check_auto_done(
        self,
        result: IsabelleSuccessResult,
        child_state: ProofState,
        all_candidates: list[ProofState],
    ) -> None:
        """Apply done tactic if we can"""
        if result.is_done or "no subgoals" not in result.result.lower():
            return None

        print("Detected 'no subgoals' - automatically adding 'done' tactic")

        done_child = self._process_command(child_state, "done", 0.0, all_candidates)

        if not done_child:
            return None

        print("AUTO-DONE: completed proof!")

        done_child.probability = child_state.probability * 0.95
        all_candidates.append(done_child)

        if done_child.is_done:
            print("*** PROOF COMPLETE (via auto-done)!")
            self._update_viz("PROOF COMPLETE (via auto-done)")

    def _remove_failed_child(self, parent_state: ProofState, child: ProofState) -> None:
        """Remove a failed child from the graph"""
        if child.state_name in parent_state.children:
            parent_state.children.remove(child.state_name)
        if child.state_name in self.graph.G:
            self.graph.G.remove_node(child.state_name)

    # if isinstance(result, IsabelleErrorResult):
    #     # failed tactic
    #     error_msg = result.error
    #
    #     # Log the error with span
    #     span.error(f"Tactic failed: {error_msg[:200]}")
    #
    #     # Regular failure handling
    #     self.graph.add_failed_tactic(parent_state.state_name, command, error_msg)
    #     print(f"  FAILED: {command}")
    #     print("  Analyzing failure...")
    #
    #     # Store the error message
    #     parent_state.failure_error_messages.append(error_msg)
    #     print(f"  Error: {error_msg}")
    #
    #     # Try statement without proof if in state mode and command has a proof method
    #     if parent_state.mode == "state":
    #         statement_without_proof = extract_statement_without_proof(command)
    #         if statement_without_proof:
    #             if statement_without_proof in [
    #                 t[0] for t in parent_state.failed_tactics
    #             ]:
    #                 print(
    #                     f"  FALLBACK SKIPPED: Statement without proof already failed: {statement_without_proof}"
    #                 )
    #             elif would_be_duplicate_statement(
    #                 statement_without_proof, parent_state, self.graph
    #             ):
    #                 print(
    #                     f"  FALLBACK SKIPPED: Would duplicate existing successful statement: {statement_without_proof}"
    #                 )
    #             else:
    #                 print(
    #                     f"\n  FALLBACK: Trying statement without proof: {statement_without_proof}"
    #                 )
    #
    #                 # Create a new child for the statement without proof
    #                 fallback_child = self.graph.add_state(
    #                     command=statement_without_proof,
    #                     parent=parent_state.state_name,
    #                     temperature=temperature,
    #                 )
    #
    #                 # Try executing the statement without proof
    #                 fallback_result = self._execute_isabelle_command(
    #                     current_state=parent_state.state_name,
    #                     command=statement_without_proof,
    #                     new_state=fallback_child.state_name,
    #                 )
    #
    #                 if isinstance(fallback_result, IsabelleSuccessResult):
    #                     # Statement without proof succeeded!
    #                     print(f"  FALLBACK SUCCESS: {statement_without_proof}")
    #
    #                     # Extract result
    #                     isabelle_result = fallback_result.result
    #                     is_done_fallback = fallback_result.is_done
    #                     self.graph.update_result(
    #                         fallback_child.state_name,
    #                         is_done_fallback,
    #                         isabelle_result,
    #                     )
    #
    #                     # Calculate probability with penalty for removing proof
    #                     proof_path = get_proof_path_to_current(
    #                         self.graph, parent_state.state_name
    #                     )
    #                     context_score = evaluate_command_in_context(
    #                         statement_without_proof,
    #                         proof_path,
    #                         parent_state.result,
    #                         isabelle_result,
    #                     )
    #
    #                     fallback_child.probability = context_score
    #
    #                     # Update mode for the fallback child (it will be in prove mode after a bare statement)
    #                     fallback_mode, fallback_lemmas = parse_isabelle_response(
    #                         isabelle_result
    #                     )
    #                     fallback_child.mode = fallback_mode
    #                     fallback_child.available_lemmas = fallback_lemmas
    #
    #                     all_candidates.append(fallback_child)
    #                     print(
    #                         f"  FALLBACK: {statement_without_proof} -> p={fallback_child.probability:.3f}"
    #                     )
    #
    #                     # Remove the original failed child but keep the fallback
    #                     parent_state.children.remove(child.state_name)
    #                     self.graph.G.remove_node(child.state_name)
    #
    #                     # Update visualization
    #                     self._update_viz(
    #                         f"Fallback: {statement_without_proof[:40]}... (p={fallback_child.probability:.3f})"
    #                     )
    #
    #                     return fallback_child
    #                 else:
    #                     # Even the statement without proof failed
    #                     print(f"  FALLBACK FAILED: {statement_without_proof}")
    #                     parent_state.children.remove(fallback_child.state_name)
    #                     self.graph.G.remove_node(fallback_child.state_name)
    #
    #     parent_state.children.remove(child.state_name)
    #     self.graph.G.remove_node(child.state_name)
    #     # update visualization after failure
    #     self._update_viz(f"Failed: {command[:40]}...")
    #     return None
    # else:
    #     # success - update result and calculate probability
    #     assert isinstance(result, IsabelleSuccessResult)
    #     is_done = result.is_done
    #     isabelle_result = result.result
    #     self.graph.update_result(child.state_name, is_done, isabelle_result)
    #
    #     # Check if we have "No subgoals!" in the result, which means we can complete with "done"
    #     if not is_done and "No subgoals!" in isabelle_result:
    #         print("  Detected 'No subgoals!' - automatically adding 'done' tactic")
    #         # Create a new child state with "done" tactic
    #         done_child = self.graph.add_state(child.state_name, "done")
    #         done_result = self._execute_isabelle_command(
    #             child.state_name, "done", done_child.state_name
    #         )
    #
    #         if isinstance(done_result, IsabelleSuccessResult):
    #             # Update the done child with the result
    #             self.graph.update_result(
    #                 done_child.state_name, done_result.is_done, done_result.result
    #             )
    #             # Set high probability for the automatic done
    #             done_child.probability = (
    #                 child.probability * 0.95
    #             )  # Slight penalty for automation
    #             all_candidates.append(done_child)
    #             print(
    #                 f"  AUTO-DONE: completed proof with p={done_child.probability:.3f}"
    #             )
    #
    #             if done_result.is_done:
    #                 print("\n  *** PROOF COMPLETE (via auto-done)! ***")
    #                 self._update_viz("PROOF COMPLETE (auto-done)!")
    #
    #     # calculate probabilities using the result after applying the command
    #     proof_path = get_proof_path_to_current(self.graph, parent_state.state_name)
    #     context_score = evaluate_command_in_context(
    #         command, proof_path, parent_state.result, isabelle_result
    #     )
    #     child.probability = context_score
    #
    #     all_candidates.append(child)
    #     print(f"  SUCCESS: {command} -> p={child.probability:.3f}")
    #     # update visualization for successful tactic
    #     self._update_viz(f"Success: {command[:40]}... (p={child.probability:.3f})")
    #
    #     if is_done:
    #         print("\n  *** PROOF COMPLETE! ***")
    #         print(f"  Final result: {isabelle_result[:100]}...")
    #         print("  No more subgoals!")
    #         # update visualization for proof completion
    #         self._update_viz("PROOF COMPLETE!")
    #
    #     return child

    def _get_mode_instructions(self, mode: str) -> str:
        """get mode-specific tactics guidance"""
        if mode == "state":
            return """
- have "P" - introduce intermediate lemma that needs proving
- have "P" by simp - prove intermediate lemma immediately
- show ?thesis - switch to proving the main goal
- show "P" - prove specific statement
- thus "P" - use 'this' (previous fact) to conclude P
- hence "P" - combines 'then' + 'thus'
- moreover - add another fact to collection
- ultimately - use all collected facts
- finally - conclude the proof
- next - move to next goal (if multiple)
- { ... } - proof block for subgoals
- assume "P" - make assumption in proof"""
        else:  # prove mode
            return """
- Basic: apply simp, apply auto, apply blast, apply force, apply fast
- Arithmetic: apply arith, apply linarith, apply algebra
- Logic: apply metis, apply meson, apply smt
- Induction: apply (induct n), apply (cases x)
- With facts: apply (simp add: assms), apply (auto simp: h1 h2)
- Advanced: apply (simp add: algebra_simps), apply (auto intro: exI)
- Completion: done (after successful apply), qed (ends proof block)
- Structured: proof -, proof (cases), proof (induct x)"""

    @lilypad.trace(name="Generate Candidate")
    async def _generate_candidates(
        self,
        parent_state: ProofState,
        temperature: float,
        self_ask_suggestion: str | None = None,
        strategic_guidance: str | None = None,
    ) -> str:
        """generates a single command candidate with specific temperature"""

        # Add metadata for this generation using a span
        with lilypad.span("Generation Metadata") as span:
            span.metadata(
                {
                    "temperature": temperature,
                    "parent_state": parent_state.state_name,
                    "depth": parent_state.depth,
                    "has_self_ask_suggestion": bool(self_ask_suggestion),
                    "has_strategic_guidance": bool(strategic_guidance),
                }
            )

        # prepare context
        mode, lemmas = parse_isabelle_response(parent_state.result)

        # Get all parent commands leading to this state
        proof_path = get_proof_path_to_current(self.graph, parent_state.state_name)
        proof_steps = ""
        if proof_path:
            proof_steps = "\nProof steps so far:\n"
            for i, cmd in enumerate(proof_path, 1):
                proof_steps += f"  {i}. {cmd}\n"

        # prepare failed summary
        failed_summary = ""
        if parent_state.failed_tactics:
            failed_summary = "\nFailed tactics (avoid these):\n```\n"
            for tactic, error in parent_state.failed_tactics[-5:]:
                failed_summary += f"- {tactic}: {error[:200]}...\n"
            failed_summary += "```"

        # prepare strategic guidance context
        strategic_context = ""
        if strategic_guidance:
            strategic_context = f"\n\nSTRATEGIC ADAPTATION:\n{strategic_guidance}\nPrioritize tactics that address the identified issues."

        # generate multiple completions with mode-specific prompt
        if mode == "state":
            # State mode prompt
            @llm.call(
                provider=PROVIDER,
                model=MODEL,
                call_params={
                    "seed": self.config.seed,
                    "temperature": temperature,
                },
            )
            def generate_state_tactic() -> str:
                # TODO: prompt einrücken oder nicht ? ruff error?

                prompt = f"""You are an expert Isabelle proof assistant in STATE MODE.

                Theorem:
                ```
                {self.theorem}
                {proof_steps}
                ```
                Current proof state:
                ```
                {parent_state.result}
                ```
                Mode: STATE (structured proof mode)
                Available lemmas:
                ```
                {lemmas}
                ```
                {failed_summary}{strategic_context}

                You are in structured proof mode. Valid commands:
                {self._get_mode_instructions("state")}

                IMPORTANT STATE MODE RULES:
                - NEVER use 'apply' commands (they are for prove mode)
                - Use 'have' for intermediate facts
                - Common patterns: 'have "... = ..." ', 'show ?thesis'

                OUTPUT REQUIREMENTS:
                - Write ONLY the Isabelle command
                - NO markdown formatting (no ```)
                - NO explanations before or after
                - NO code blocks
                - Just the pure command like: have "x = 5"

                Generate EXACTLY ONE Isabelle command for state mode:
                """

                # Print the full prompt being sent to LLM
                print("\n" + "=" * 80)
                print("LLM PROMPT (STATE MODE):")
                print("=" * 80)
                print(prompt)
                print("=" * 80 + "\n")

                return prompt

            response = generate_state_tactic()
        else:
            # Prove mode prompt
            @llm.call(
                provider=PROVIDER,
                model=MODEL,
                call_params={
                    "seed": self.config.seed,
                    "temperature": temperature,
                },
            )
            def generate_prove_tactic() -> str:
                # TODO: prompt einrücken oder nicht ? ruff error?
                prompt = f"""You are an expert Isabelle proof assistant in PROVE MODE.

                Theorem:
                ```
                {self.theorem}
                {proof_steps}
                ```
                Current proof state:
                ```
                {parent_state.result}
                ```
                Mode: PROVE (tactical proof mode)
                Available lemmas:
                ```
                {lemmas}
                ```
                {failed_summary}{strategic_context}

                You are in tactical proof mode. Valid commands:
                {self._get_mode_instructions("prove")}

                IMPORTANT PROVE MODE RULES:
                - Use 'apply' for step-by-step tactics
                - NEVER use 'have', 'show', 'fix' (they are for state mode)
                - Can switch to state mode with 'proof -'
                - Make use of existing assumptions (assms) and pre-defined lemmas
                - When you have named assumptions (like h0, h1 in "assumes h0: ... and h1: ..."), you can:
                  * Use them individually: "using h0 apply simp", "using h1 apply auto"
                  * Use them together: "using h0 h1 apply simp", "using h0 h1 apply linarith"
                  * Add them to tactics: "apply (simp add: h0 h1)", "apply (auto simp: h0 h1)"
                - Common patterns: "using assms apply auto", "apply (auto simp add: ...)", "apply blast"

                OUTPUT REQUIREMENTS:
                - Write ONLY the Isabelle command
                - NO markdown formatting (no ```)
                - NO explanations before or after
                - NO code blocks
                - Just the pure command like: apply simp

                Generate EXACTLY ONE Isabelle command for prove mode:
                """

                # Print the full prompt being sent to LLM
                print("\n" + "=" * 80)
                print("LLM PROMPT (PROVE MODE):")
                print("=" * 80)
                print(prompt)
                print("=" * 80 + "\n")

                return prompt

            response = generate_prove_tactic()

        # extract command from response
        command = response.content.strip()

        # Clean up common markdown formatting
        if command.startswith("```"):
            # Extract content between code blocks
            lines = command.split("\n")
            cleaned_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                elif in_block:
                    cleaned_lines.append(line)
            if cleaned_lines:
                command = "\n".join(cleaned_lines).strip()
        # Remove any remaining backticks
        command = command.replace("`", "").strip()

        return command

    # @lilypad.trace(name="Beam Search Level")
    # async def beam_search_level(
    #     self,
    #     current_beam: list[ProofState],
    #     max_depth: int,
    # ) -> ProofState | None:
    #     """execute one level of beam search"""
    #
    #     # Add metadata for this beam search level using a span
    #     with lilypad.span("Beam Level Metadata") as span:
    #         span.metadata(
    #             {
    #                 "depth": self.current_depth,
    #                 "beam_size": len(current_beam),
    #                 "max_depth": max_depth,
    #             }
    #         )
    #
    #     if not current_beam or self.current_depth >= max_depth:
    #         # Before giving up, check if there are pending sledgehammer results
    #         # TODO: rewrite to use sledgehammer manager
    #         if self.graph.sledgehammer_futures:
    #             print(
    #                 f"\nWaiting for {len(self.graph.sledgehammer_futures)} pending sledgehammer results..."
    #             )
    #
    #             # Wait for all pending sledgehammer tasks with timeout
    #             # Create a mapping of futures to state names
    #             future_to_state = {}
    #             for state_name, future in self.graph.sledgehammer_futures.items():
    #                 future_to_state[future] = state_name
    #
    #             pending_futures = list(self.graph.sledgehammer_futures.values())
    #
    #             try:
    #                 # TODO: check sledgehammer timeout settings
    #                 done, pending = await asyncio.wait(pending_futures, timeout=80.0)
    #                 print(
    #                     f"{len(done)} sledgehammer tasks completed, {len(pending)} timed out"
    #                 )
    #             except Exception as e:
    #                 print(f"Error waiting for sledgehammer: {e}")
    #
    #             # Check if any sledgehammer found proofs
    #             sledgehammer_succeeded = False
    #             for future in done:
    #                 try:
    #                     results = future.result()
    #                     state_name = future_to_state.get(future)
    #                     if results and state_name:
    #                         print(
    #                             f"\nDEBUG: Processing sledgehammer results for {state_name}"
    #                         )
    #                         print(f"  Raw results: {results!r}")
    #                         # Parse sledgehammer output to extract tactic
    #                         tactic = parse_sledgehammer_output(results)
    #                         print(f"  Parsed tactic: {tactic!r}")
    #                         if tactic:
    #                             print(
    #                                 f"\nSledgehammer found tactic for {state_name}: {tactic}"
    #                             )
    #                             # Find the state in our graph
    #                             state = self.graph.get_state(state_name)
    #                             if state and not state.is_done:
    #                                 # Try the sledgehammer tactic
    #                                 print(f"  Trying sledgehammer tactic: {tactic}")
    #                                 # Process the sledgehammer tactic
    #                                 child = self._process_candidate(
    #                                     parent_state=state,
    #                                     command=tactic,
    #                                     temperature=0.0,  # Sledgehammer tactics are deterministic
    #                                     all_candidates=[],  # We're not adding to beam, just checking
    #                                 )
    #                                 if child and child.is_done:
    #                                     print(
    #                                         f"Sledgehammer tactic succeeded: {tactic}"
    #                                     )
    #                                     sledgehammer_succeeded = True
    #                                     return child
    #                         else:
    #                             print("  No tactic extracted from sledgehammer output")
    #                     else:
    #                         if not results:
    #                             print(
    #                                 f"\nDEBUG: Sledgehammer returned None/empty for {state_name}"
    #                             )
    #                         if not state_name:
    #                             print("\nDEBUG: No state_name mapping for future")
    #                 except Exception as e:
    #                     print(f"Error processing sledgehammer result: {e}")
    #
    #             if not sledgehammer_succeeded:
    #                 print("No sledgehammer tactics succeeded")
    #
    #         return None
    #
    #     all_candidates = []
    #
    #     # generate candidates for each node in beam
    #     for parent_state in current_beam:
    #         if parent_state.is_done:
    #             # keep completed proofs in beam
    #             all_candidates.append(parent_state)
    #             continue
    #
    #         print(
    #             f"\nexploring from {parent_state.state_name} (p={parent_state.probability:.3f})"
    #         )
    #
    #         # Check for failure patterns and adapt strategy if needed
    #         strategic_guidance = None
    #         if parent_state.has_failure_pattern(min_occurrences=3):
    #             print("\nFAILURE PATTERN DETECTED - Reflecting on strategy...")
    #
    #             pattern_type = "repeated errors"
    #
    #             # Format failure summary
    #             failure_summary = "\n".join(
    #                 [
    #                     f"- {error_msg[:100]}..."
    #                     for error_msg in parent_state.failure_error_messages[-5:]
    #                 ]
    #             )
    #
    #             print(f"Pattern type: {pattern_type}")
    #             print(f"Total failures: {len(parent_state.failure_error_messages)}")
    #
    #             # Get strategic reflection
    #             strategic_reflection = reflect_on_failures(
    #                 theorem=self.theorem,
    #                 proof_state=parent_state.result,
    #                 pattern_type=pattern_type,
    #                 failure_summary=failure_summary,
    #             )
    #
    #             strategic_guidance = strategic_reflection.content
    #             print(f"\nStrategic guidance:\n{strategic_guidance}\n")
    #
    #         # Try automated tools first
    #         automated_tactics = []
    #
    #         # Parse current mode for tactic selection
    #         mode, lemmas = parse_isabelle_response(parent_state.result)
    #         parent_state.mode = mode
    #         parent_state.available_lemmas = lemmas
    #
    #         # Track commands already tried from this state
    #         tried_commands = set()
    #
    #         # Add already failed tactics to tried commands
    #         for failed_tactic, _ in parent_state.failed_tactics:
    #             tried_commands.add(failed_tactic)
    #
    #         # Add already successful children to tried commands
    #         for child_name in parent_state.children:
    #             child_state = self.graph.get_state(child_name)
    #             tried_commands.add(child_state.command)
    #
    #         # 1. Quick tactic attempts (replaces try0)
    #         if self.config.use_quick_tactics:
    #             print("\nTrying quick tactics...")
    #             quick_tactics_tried = 0
    #             quick_tactics_succeeded = 0
    #
    #             # Select tactics based on mode
    #             if mode == "state":
    #                 # In state mode, we can only make statements, not prove them
    #                 # Quick tactics don't apply here - we need to generate statements
    #                 quick_tactics = []
    #             else:  # prove mode
    #                 quick_tactics = list(self.config.quick_tactics)
    #
    #                 # if theorem has assumptions, add tactics that use them
    #                 if "assumes" in self.theorem:
    #                     quick_tactics.extend(
    #                         [
    #                             "apply (simp add: assms)",
    #                             "apply (auto simp add: assms)",
    #                             "using assms apply simp",
    #                             "using assms apply auto",
    #                             "using assms apply arith",
    #                             "using assms apply linarith",
    #                             "using assms apply blast",
    #                             "using assms apply fastforce",
    #                         ]
    #                     )
    #
    #             for tactic in quick_tactics:
    #                 if tactic in tried_commands:
    #                     continue
    #
    #                 quick_tactics_tried += 1
    #
    #                 with lilypad.span("Execute Tactic") as span:
    #                     span.metadata(
    #                         {
    #                             "command": tactic,
    #                             "parent_state": parent_state.state_name,
    #                             "source": "quick_tactic",
    #                         }
    #                     )
    #
    #                     try:
    #                         temp_state = f"quick_{parent_state.state_name}_{tactic.replace(' ', '_')}"
    #                         is_done, result = self.session.execute(
    #                             parent_state.state_name, tactic, temp_state
    #                         )
    #                         # If the tactic succeeds (doesn't throw an error), add it
    #                         # regardless of whether it completes the proof
    #                         automated_tactics.append((tactic, None, "quick"))
    #                         quick_tactics_succeeded += 1
    #                         print(
    #                             f"  Quick tactic works: {tactic} (completes proof: {is_done})"
    #                         )
    #                         # Only break if we completely solved the goal
    #                         if is_done:
    #                             break
    #                     except QIsabelleServerError as e:
    #                         span.error(f"Quick tactic failed: {str(e)[:200]}")
    #                         continue
    #
    #             # log quick tactics performance
    #             if quick_tactics_tried > 0:
    #                 pass
    #
    #         # 2. Start sledgehammer in background if conditions are met
    #         if (
    #             self.config.use_sledgehammer
    #             and mode == "prove"  # sledgehammer only works in prove mode
    #         ):
    #             # start sledgehammer async (non-blocking)
    #             await self.graph.start_sledgehammer_async(
    #                 parent_state.state_name, self.session
    #             )
    #             print(
    #                 f"  Started sledgehammer in background for {parent_state.state_name}"
    #             )
    #
    #         # 3. Check for completed sledgehammer results (from this or previous iterations)
    #         if mode == "prove":
    #             sledgehammer_results = self.graph.check_sledgehammer_result(
    #                 parent_state.state_name
    #             )
    #             if sledgehammer_results is not None:  # results are ready
    #                 print(
    #                     f"\n  Sledgehammer results ready: found {len(sledgehammer_results)} tactics"
    #                 )
    #                 for tactic in sledgehammer_results[:3]:  # Top 3 suggestions
    #                     if tactic not in tried_commands:
    #                         automated_tactics.append((tactic, None, "sledgehammer"))
    #                         print(f"  Sledgehammer suggests: {tactic}")
    #
    #         # Process automated tactics with high priority
    #         for tactic, _, source in automated_tactics:
    #             if tactic not in tried_commands:
    #                 child = self._process_candidate(
    #                     parent_state,
    #                     tactic,
    #                     0.0,  # temperature 0 for deterministic
    #                     all_candidates,
    #                 )
    #                 if child:
    #                     # Context score already calculated in _process_candidate
    #                     print(
    #                         f"  {source.upper()}: {tactic} -> p={child.probability:.3f}"
    #                     )
    #                     tried_commands.add(tactic)
    #
    #         # try with self-ask reasoning if no automated tactics succeeded
    #         mode, lemmas = parse_isabelle_response(parent_state.result)
    #
    #         # Get proof path for this state
    #         proof_path = get_proof_path_to_current(self.graph, parent_state.state_name)
    #
    #         # Build the query string
    #         query_parts = [
    #             f"Theorem: {self.theorem}",
    #         ]
    #
    #         # Add proof steps if any
    #         if proof_path:
    #             query_parts.append("Proof steps so far:")
    #             for i, cmd in enumerate(proof_path, 1):
    #                 query_parts.append(f"  {i}. {cmd}")
    #
    #         query_parts.extend(
    #             [
    #                 f"Current goals: {parent_state.result}",
    #                 f"Current mode: {mode}",
    #             ]
    #         )
    #
    #         # Add optional components
    #         if mode == "state":
    #             query_parts[0] = (
    #                 f"Current mode: {mode}"  # Put mode first for state mode
    #             )
    #             if "proof (state)" in parent_state.result:
    #                 query_parts.append("proof (state)")
    #
    #         if lemmas:
    #             query_parts.append(f"Available lemmas: {', '.join(lemmas)}")
    #
    #         if parent_state.failed_tactics:
    #             # Include last 5 failed tactics with shortened error messages
    #             failed_str = str(
    #                 [
    #                     (t, e[:50] + "..." if len(e) > 50 else e)
    #                     for t, e in parent_state.failed_tactics[-5:]
    #                 ]
    #             )
    #             query_parts.append(f"Failed tactics: {failed_str}")
    #
    #         query = "\n".join(query_parts)
    #
    #         # Format examples - limit to most relevant ones
    #         # Filter examples based on current mode
    #         relevant_examples = [
    #             ex
    #             for ex in few_shot_examples
    #             if f"mode: {mode}" in ex["question"].lower()
    #             or (mode == "prove" and "mode: state" not in ex["question"].lower())
    #         ][
    #             :3
    #         ]  # Take at most 3 examples
    #
    #         examples_str = "\n\n".join(
    #             [f"Q: {ex['question']}\nA: {ex['answer']}" for ex in relevant_examples]
    #         )
    #
    #         # Build failure context from error messages
    #         failure_context = (
    #             "\n".join(parent_state.failure_error_messages[-5:])
    #             if parent_state.failure_error_messages
    #             else ""
    #         )
    #
    #         print(
    #             "\nLLM API Call: self_ask_reasoning (temperature=N/A - using default)"
    #         )
    #         print("\nSELF-ASK QUERY:")
    #         print("=" * 80)
    #         print(query)
    #         print("=" * 80 + "\n")
    #
    #         reasoning_response = self_ask_reasoning(
    #             theorem=self.theorem,
    #             proof_state=parent_state.result,
    #             mode=mode,
    #             available_lemmas=lemmas,
    #             failed_tactics=parent_state.failed_tactics,
    #             examples=examples_str,
    #             query=query,
    #             failure_context=failure_context,
    #             proof_path=proof_path,
    #         )
    #
    #         # Extract the final tactic from the response
    #         content = reasoning_response.content
    #
    #         # Print full self-ask reasoning
    #         print(f"\nSelf-ask reasoning:\n{content}\n")
    #
    #         # Look for the standard self-ask format
    #         if "So the final answer is:" in content:
    #             suggested_tactic = content.split("So the final answer is:")[-1].strip()
    #
    #             # Clean up the suggested tactic
    #             # Remove markdown code blocks if present
    #             if suggested_tactic.startswith("```"):
    #                 # Extract content between code blocks
    #                 lines = suggested_tactic.split("\n")
    #                 # Find the first line that's not a code block marker
    #                 tactic_lines = []
    #                 in_block = False
    #                 for line in lines:
    #                     if line.startswith("```"):
    #                         in_block = not in_block
    #                     elif in_block and line.strip():
    #                         tactic_lines.append(line.strip())
    #                 # Take only the first command if multiple were given
    #                 if tactic_lines:
    #                     suggested_tactic = tactic_lines[0]
    #                 else:
    #                     suggested_tactic = None
    #             else:
    #                 # If multi-line, take only the first line
    #                 lines = suggested_tactic.strip().split("\n")
    #                 suggested_tactic = lines[0].strip()
    #
    #             # Remove any remaining markdown or formatting
    #             if suggested_tactic:
    #                 suggested_tactic = suggested_tactic.replace("`", "").strip()
    #         else:
    #             # Fallback: take the last line if format is different
    #             lines = content.strip().split("\n")
    #             suggested_tactic = lines[-1].strip()
    #             if suggested_tactic.startswith("```"):
    #                 suggested_tactic = None
    #
    #         print(f"self-ask suggested: {suggested_tactic}")
    #
    #         # update visualization for self-ask suggestion
    #         if suggested_tactic:
    #             self._update_viz(f"Self-ask suggests: {suggested_tactic[:50]}...")
    #
    #         # First, try the self-ask suggested tactic if not already tried
    #         if (
    #             suggested_tactic
    #             and suggested_tactic not in tried_commands
    #             and is_valid_isabelle_command(suggested_tactic)
    #         ):
    #             print(f"\nTrying self-ask suggestion: {suggested_tactic}")
    #             tried_commands.add(suggested_tactic)
    #
    #             # Create state for self-ask suggestion
    #             child = self.graph.add_state(
    #                 command=suggested_tactic,
    #                 parent=parent_state.state_name,
    #                 temperature=0.0,  # Mark as deterministic
    #             )
    #
    #             # Execute the self-ask suggested tactic
    #             with lilypad.span("Execute Tactic") as span:
    #                 span.metadata(
    #                     {
    #                         "command": suggested_tactic,
    #                         "parent_state": parent_state.state_name,
    #                         "source": "self_ask",
    #                     }
    #                 )
    #
    #                 result = self._execute_isabelle_command(
    #                     current_state=parent_state.state_name,
    #                     command=suggested_tactic,
    #                     new_state=child.state_name,
    #                 )
    #
    #             if isinstance(result, IsabelleErrorResult):
    #                 # Failed tactic
    #                 error_msg = result.error
    #
    #                 # Log the error with span
    #                 span.error(f"Tactic failed: {error_msg[:200]}")
    #
    #                 # Regular failure handling
    #                 self.graph.add_failed_tactic(
    #                     parent_state.state_name, suggested_tactic, error_msg
    #                 )
    #
    #                 # Analyze the self-ask failure
    #                 print(f"  FAILED: {suggested_tactic}")
    #                 print("  Analyzing self-ask failure...")
    #
    #                 # Store the error message
    #                 parent_state.failure_error_messages.append(error_msg)
    #                 print(f"  Error: {error_msg}")
    #
    #                 parent_state.children.remove(child.state_name)
    #                 self.graph.G.remove_node(child.state_name)
    #             else:
    #                 # Success - update result and calculate probability
    #                 assert isinstance(result, IsabelleSuccessResult)
    #                 is_done = result.is_done
    #                 isabelle_result = result.result
    #                 self.graph.update_result(child.state_name, is_done, isabelle_result)
    #
    #                 # For self-ask, calculate context score
    #                 proof_path = get_proof_path_to_current(
    #                     self.graph, parent_state.state_name
    #                 )
    #                 context_score = evaluate_command_in_context(
    #                     suggested_tactic,
    #                     proof_path,
    #                     parent_state.result,
    #                     isabelle_result,
    #                 )
    #                 child.probability = context_score
    #
    #                 all_candidates.append(child)
    #                 print(f"  SUCCESS: {suggested_tactic} -> p={child.probability:.3f}")
    #
    #                 if is_done:
    #                     print("  PROOF COMPLETE!")
    #                     print(f"  Final result: {isabelle_result[:100]}...")
    #
    #         # Generate candidates with different temperatures - TWICE
    #         # First pass: WITH self-ask context
    #         print("\n--- Temperature-based generation WITH self-ask context ---")
    #         self._update_viz("Generating candidates with self-ask context...")
    #         for temperature in self.config.temperatures:
    #             try:
    #                 print(
    #                     f"\nLLM API Call: generate_tactics with self-ask (temperature={temperature})"
    #                 )
    #                 command, temp = await self._generate_candidates(
    #                     parent_state,
    #                     temperature,
    #                     self_ask_suggestion=suggested_tactic,
    #                     strategic_guidance=strategic_guidance,
    #                 )
    #
    #                 print(f"  Generated candidate: {command}")
    #
    #                 # skip if already tried or invalid
    #                 if command in tried_commands:
    #                     print(f"  SKIPPING (already tried): {command}")
    #                     continue
    #                 if not is_valid_isabelle_command(command):
    #                     print(f"  SKIPPING (invalid command): {command}")
    #                     continue
    #
    #                 # Mark as tried
    #                 tried_commands.add(command)
    #
    #                 # Process candidate
    #                 child = self._process_candidate(
    #                     parent_state,
    #                     command,
    #                     temp,
    #                     all_candidates,
    #                 )
    #
    #             except Exception as e:
    #                 import traceback
    #
    #                 print(f"  WARNING: error generating candidates: {e!s}")
    #                 print(f"  Traceback: {traceback.format_exc()}")
    #
    #         # Second pass: WITHOUT self-ask context (original behavior)
    #         print("\n--- Temperature-based generation WITHOUT self-ask context ---")
    #         self._update_viz("Generating candidates without self-ask context...")
    #         for temperature in self.config.temperatures:
    #             try:
    #                 print(
    #                     f"\nLLM API Call: generate_tactics without self-ask (temperature={temperature})"
    #                 )
    #                 command, temp = await self._generate_candidates(
    #                     parent_state, temperature, strategic_guidance=strategic_guidance
    #                 )
    #
    #                 print(f"  Generated candidate: {command}")
    #
    #                 # skip if already tried or invalid
    #                 if command in tried_commands:
    #                     print(f"  SKIPPING (already tried): {command}")
    #                     continue
    #                 if not is_valid_isabelle_command(command):
    #                     print(f"  SKIPPING (invalid command): {command}")
    #                     continue
    #
    #                 # Mark as tried
    #                 tried_commands.add(command)
    #
    #                 # Process candidate
    #                 child = self._process_candidate(
    #                     parent_state,
    #                     command,
    #                     temp,
    #                     all_candidates,
    #                 )
    #
    #             except Exception as e:
    #                 import traceback
    #
    #                 print(f"  WARNING: error generating candidates: {e!s}")
    #                 print(f"  Traceback: {traceback.format_exc()}")
    #
    #         # update visualization after processing candidates from this parent
    #         candidates_from_parent = [
    #             c for c in all_candidates if c.parent == parent_state.state_name
    #         ]
    #         if candidates_from_parent:
    #             self._update_viz(
    #                 f"Processed {len(candidates_from_parent)} candidates from {parent_state.state_name}"
    #             )
    #
    #     # select top k candidates for next beam
    #     all_candidates.sort(key=lambda x: x.probability, reverse=True)
    #     next_beam = all_candidates[: self.config.beam_width]
    #
    #     # update beam ranks
    #     for i, state in enumerate(next_beam):
    #         state.beam_rank = i + 1
    #
    #     # record beam history
    #     self.graph.beam_history.append([s.state_name for s in next_beam])
    #
    #     # visualize
    #     self._update_viz(f"Beam Search Level {self.current_depth}")
    #
    #     # check for completed proofs
    #     for state in next_beam:
    #         if state.is_done:
    #             return state
    #
    #     # move to next level
    #     self.current_depth += 1
    #
    #     return await self.beam_search_level(next_beam, max_depth)

    # @lilypad.trace(name="Beam Search Level")
    # async def run_beam_search(
    #     self, current_beam: list[ProofState], max_depth: int,
    # ) -> ProofState | None:
    #     """Run beam search from start state up to max_depth
    #
    #     Returns:
    #         Completed proof state if found, None if otherwiese
    #     """
    #
    #     print("Starting beam search with config:")
    #     print(f"    beam width: {self.config.beam_width}")
    #     print(f"    temperatures: {self.config.temperatures}")
    #     print(f"    max depth: {max_depth}")
    #
    #     # init search
    #     self.current_depth = 0
    #     current_beam = [start_state]
    #
    #     while current_beam and self.current_depth < max_depth:
    #
    #
    #     return None

    # async def search(
    #     self,
    #     start_state: ProofState,
    #     max_depth: int = 10,
    # ) -> ProofState | None:
    #     """run beam search from start state"""
    #
    #     print("\nStarting beam search with config:")
    #     print(f"  beam width: {self.config.beam_width}")
    #     print(f"  temperatures: {self.config.temperatures}")
    #     print(f"  max depth: {max_depth}")
    #
    #     self.current_depth = 0
    #     return await self.beam_search_level([start_state], max_depth)

    # TODO: add lilypad trace to this as well, maybe with logging the start_state or problem name?
    @lilypad.trace("Beam Search")
    async def search(
        self,
        start_state: ProofState,
        max_depth: int,
    ) -> ProofState | None:
        """Run beam search from start state up to max_depth

        Returns:
            Completed proof state if found, None if otherwiese
        """

        print("Starting beam search with config:")
        print(f"    beam width: {self.config.beam_width}")
        print(f"    temperatures: {self.config.temperatures}")
        print(f"    max depth: {max_depth}")

        # init search
        self.current_depth = 0
        current_beam = [start_state]

        while current_beam and self.current_depth < max_depth:
            print(f"\n{'=' * 60}")
            print(f"BEAM SEARCH LEVEL {self.current_depth}")
            print(f"{'=' * 60}")

            next_beam, completed_proof = await self._process_single_beam_level(
                current_beam
            )

            if completed_proof:
                return completed_proof

            # Always check for sledgehammer results, even if beam is empty
            sledgehammer_states = await self._process_remaining_sledgehammer()

            # Check if any sledgehammer found a complete proof
            completed_proofs = [s for s in sledgehammer_states if s.is_done]
            partial_proofs = [s for s in sledgehammer_states if not s.is_done]

            if completed_proofs:
                # Return the best completed proof
                best_complete = max(completed_proofs, key=lambda s: s.probability)
                print(
                    f"\nSledgehammer completed the proof with: {best_complete.command[:80]}..."
                )
                return best_complete

            if partial_proofs:
                print(
                    f"\nSledgehammer made partial progress on {len(partial_proofs)} states"
                )

            # Add successful sledgehammer states to the beam
            if sledgehammer_states:
                # If beam is empty, use sledgehammer states as the new beam
                if not next_beam:
                    print(
                        f"\nBeam was empty, using {len(sledgehammer_states)} sledgehammer results as new beam"
                    )
                    next_beam = sledgehammer_states[: self.config.beam_width]
                else:
                    # Combine with existing beam and re-select top k
                    all_states = list(next_beam) + sledgehammer_states
                    all_states.sort(key=lambda x: x.probability, reverse=True)
                    next_beam = all_states[: self.config.beam_width]

                    # Log which sledgehammer states made it into the beam
                    sledge_in_beam = [s for s in next_beam if s in sledgehammer_states]
                    print(
                        f"\nAdded {len(sledgehammer_states)} sledgehammer results, {len(sledge_in_beam)} made it into beam"
                    )

            # move to the next level
            current_beam = next_beam
            self.current_depth += 1

        print(f"Search done and no proof found at depth {self.current_depth}")
        return None

    @lilypad.trace("Single Beam Level")
    async def _process_single_beam_level(
        self, current_beam: list[ProofState]
    ) -> tuple[list[ProofState], ProofState | None]:
        """Process a single level of beam search

        Returns:
            Tuple of (next_beam, completed_proof)
            - next_beam: states for the next level
            - completed_proof: a state that completes the proof if found, None otherwise

        """
        all_candidates = []

        for parent_state in current_beam:
            if parent_state.is_done:
                all_candidates.append(parent_state)
                continue

            print(
                f"\nExploring from {parent_state.state_name} (p={parent_state.probability:.3f})"
            )

            parent_candidates = await self._explore_from_state(parent_state)
            all_candidates.extend(parent_candidates)

            for c in parent_candidates:
                if c.is_done:
                    print(f"\nFound complete proof at {c.state_name}")
                    return [], c

        next_beam = self._select_next_beam(all_candidates)

        self._update_viz(f"Beam Search Level {self.current_depth} completed")

        return next_beam, None

    async def _explore_from_state(self, parent_state: ProofState) -> list[ProofState]:
        """Explore all candidates from a single parent state


        Returns:
            List of successful child states
        """

        candidates = []

        # 1. check for failure patterns
        strategic_guidance = self._analyze_failure_patterns(parent_state)

        # 2. parse current mode
        mode, lemmas = parse_isabelle_response(parent_state.result)
        parent_state.mode = mode
        parent_state.available_lemmas = lemmas

        # 3. track tried commands
        tried_commands = self._get_tried_commands(parent_state)

        # 4. try automated tactics first
        automated_candidates = await self._try_automated_tactics(
            parent_state, mode, tried_commands
        )
        candidates.extend(automated_candidates)

        # 5. start sledgehammer if applicable
        if self.config.use_sledgehammer and mode == "prove":
            await self.sledgehammer_manager.start_sledgehammer_async(
                parent_state.state_name,
            )

        # 5b. check if any sledgehammer results are ready for this state
        sledgehammer_result = self.sledgehammer_manager.check_sledgehammer_result(
            parent_state.state_name
        )
        if sledgehammer_result is not None and sledgehammer_result:
            # Sledgehammer found a tactic
            for tactic in sledgehammer_result:
                if tactic not in tried_commands:
                    print(f"  Sledgehammer ready with tactic: {tactic[:50]}...")
                    child = self._process_command(parent_state, tactic, 0.0, candidates)
                    if child:
                        # Boost probability for sledgehammer tactics since they're proven to work
                        child.probability = min(1.0, child.probability * 1.5)
                        print(
                            f"  SLEDGEHAMMER: {tactic[:50]}... -> p={child.probability:.3f}"
                        )
                        tried_commands.add(tactic)

        # 6. try self-ask reasoning
        self_ask_candidate = await self._try_self_ask_reasoning(
            parent_state, tried_commands
        )
        if self_ask_candidate:
            candidates.append(self_ask_candidate)

        # 7. generate candidates based on different temperatures
        temperature_candidates = await self._generate_temperature_candidates(
            parent_state, tried_commands, self_ask_candidate, strategic_guidance
        )
        candidates.extend(temperature_candidates)

        if candidates:
            self._update_viz(
                f"Generated {len(candidates)} candidates from {parent_state.state_name}"
            )

        return candidates

    async def _try_automated_tactics(
        self, parent_state: ProofState, mode: str, tried_commands: set[str]
    ) -> list[ProofState]:
        """Try quick automated tactics

        Returns:
            List of successful candidates from automated tactics
        """
        candidates = []

        if self.config.use_quick_tactics and mode == "prove":
            quick_tactics = self._get_quick_tactics_for_theorem()

            for t in quick_tactics:
                if t in tried_commands:
                    continue

                if self._test_quick_tactic(parent_state, t):
                    child = self._process_command(parent_state, t, 0.0, candidates)
                    if child:
                        print(f"    QUICK: {t} -> p={child.probability:.3f}")
                        tried_commands.add(t)

                        if child.is_done:
                            break
        return candidates

    async def _generate_temperature_candidates(
        self,
        parent_state: ProofState,
        tried_commands: set[str],
        self_ask_suggestion: ProofState | None,
        strategic_guidance: str | None,
    ) -> list[ProofState]:
        """Generate candidates using different temperatures

        Returns:
            List of successful candidats from temperature generation
        """
        candidates = []

        # get self-ask suggestion text if available
        suggestion_text = None
        if self_ask_suggestion:
            suggestion_text = self_ask_suggestion.command

        # two passes: with and w/o self-ask context
        for use_self_ask in [True, False]:
            print(
                f"\n--- Temperature generation, self-ask context included: {use_self_ask}"
            )

            for temperature in self.config.temperatures:
                try:
                    command = await self._generate_candidates(
                        parent_state=parent_state,
                        temperature=temperature,
                        self_ask_suggestion=suggestion_text if use_self_ask else None,
                        strategic_guidance=strategic_guidance,
                    )

                    if command in tried_commands or not is_valid_isabelle_command(
                        command
                    ):
                        continue

                    tried_commands.add(command)

                    child = self._process_command(
                        parent_state=parent_state,
                        command=command,
                        temperature=temperature,
                        all_candidates=candidates,
                    )

                    if child:
                        print(f"    Generated: {child} -> p={child.probability:.3f}")

                except Exception as e:
                    print(f"    WARNING: Generation error: {e}")

        return candidates

    async def _process_remaining_sledgehammer(self) -> list[ProofState]:
        """Process all pending sledgehammer results at end of level

        Returns:
            List of successful proof states from sledgehammer
        """

        print(
            f"\nWaiting for {self.sledgehammer_manager.get_pending_count()} sledgehammer results..."
        )

        results = await self.sledgehammer_manager.wait_for_all_results(timeout=80.0)

        successful_states = []

        for goal_state, command in results.items():
            if not command:
                continue

            print(f"\nTrying sledgehammer command for {goal_state}: {command}")

            child = self._process_command(
                parent_state=self.graph.get_state(goal_state),
                command=command,
                temperature=0.0,
                all_candidates=[],
            )

            if not child:
                continue

            # Extract info about remaining goals
            remaining_goals = "unknown"
            if "goal" in child.result.lower():
                # Try to extract goal count
                import re

                goal_match = re.search(r"goal \((\d+) subgoal", child.result)
                if goal_match:
                    remaining_goals = f"{goal_match.group(1)} subgoals"
                elif "No subgoals!" in child.result:
                    remaining_goals = "no subgoals"

            print(
                f"Sledgehammer tactic applied, remaining: {remaining_goals}, done: {child.is_done}"
            )

            # Boost probability for sledgehammer tactics since they're proven to work
            child.probability = min(1.0, child.probability * 1.5)
            successful_states.append(child)

        # Log summary of sledgehammer results
        if successful_states:
            print(
                f"\nSledgehammer summary: {len(successful_states)} successful tactics found"
            )
            for state in successful_states:
                print(
                    f"  - {state.state_name}: {state.command[:50]}... (done: {state.is_done}, p={state.probability:.3f})"
                )

        return successful_states

    def _select_next_beam(self, candidates: list[ProofState]) -> list[ProofState]:
        """select top k candidates for next beam"""
        candidates.sort(key=lambda x: x.probability, reverse=True)

        next_beam = candidates[: self.config.beam_width]

        for i, state in enumerate(next_beam):
            state.beam_rank = i + 1

        self.graph.beam_history.append([s.state_name for s in next_beam])

        print(f"\nSelected {len(next_beam)} states for next beam:")
        for state in next_beam:
            print(f"  - {state.state_name}: p={state.probability:.3f}")

        return next_beam

    def _get_tried_commands(self, parent_state: ProofState) -> set[str]:
        """get all commands already tried from this state, so failed and successful ones"""
        tried = set()

        for tactic, _ in parent_state.failed_tactics:
            tried.add(tactic)

        for child_name in parent_state.children:
            tried.add(self.graph.get_state(child_name).command)

        return tried

    def _analyze_failure_patterns(self, parent_state: ProofState) -> str | None:
        """Analyze failure patterns and get strategic guidance if needed"""
        if not parent_state.has_failure_pattern(min_occurrences=3):
            return None

        print("\nFAILURE PATTERN DETECTED - Reflecting on strategy...")

        failure_summary = "\n".join(
            [
                f"- {error[:200]}..."
                for error in parent_state.failure_error_messages[-5:]
            ]
        )

        reflection = reflect_on_failures(
            theorem=self.theorem,
            proof_state=parent_state.result,
            pattern_type="repeated errors",
            failure_summary=failure_summary,
        )

        print(f"Strategic guidance: {reflection.content}")
        return reflection.content

    def _get_quick_tactics_for_theorem(self) -> list[str]:
        """Get quick tactics to try based on theorem structure"""
        tactics = list(self.config.quick_tactics)

        if "assumes" in self.theorem:
            tactics.extend(
                [
                    "apply (simp add: assms)",
                    "apply (auto simp add: assms)",
                    "using assms apply simp",
                    "using assms apply auto",
                    "using assms apply blast",
                ]
            )

        return tactics

    def _test_quick_tactic(self, parent_state: ProofState, tactic: str) -> bool:
        """Test if a quick tactic works (doesn't throw error)"""
        try:
            temp_state = f"quick_test_{parent_state.state_name}_{hash(tactic)}"
            self.session.execute(parent_state.state_name, tactic, temp_state)
            return True
        except QIsabelleServerError:
            return False

    async def _try_self_ask_reasoning(
        self, parent_state: ProofState, tried_commands: set[str]
    ) -> ProofState | None:
        """Try self-ask reasoning to generate a command

        Returns:
            Successful child state if tactic works, None otherwise
        """
        mode, lemmas = parse_isabelle_response(parent_state.result)
        mode_description = (
            "structured proof mode" if mode == "state" else "tactical proof mode"
        )

        # construct a description about the proof history
        proof_path = get_proof_path_to_current(self.graph, parent_state.state_name)
        if not proof_path:
            proof_history_description = "No steps yet - this is the first tactic."
        lines = []
        for i, cmd in enumerate(proof_path, 1):
            lines.append(f"{i}. {cmd}")
        proof_history_description = "\n".join(lines)

        # construct a description about failed attempts
        if not parent_state.failed_tactics:
            failed_attempts_description = "No failed attempts yet."
        lines = []
        for tactic, error in parent_state.failed_tactics[-5:]:  # Last 5 failures
            error_short = error[:100] + "..." if len(error) > 100 else error
            lines.append(f"- {tactic}\n  Error: {error_short}")
        failed_attempts_description = "\n".join(lines)

        available_lemmas = ", ".join(lemmas) if lemmas else "none"

        # description about instructions for the current mode
        #         if mode == "state":
        #             mode_instructions = """Valid commands in STATE MODE:
        # - have "statement" - State a fact to prove later
        # - have "statement" by method - State and prove immediately
        # - show ?thesis by method - Prove the main goal
        # - show "statement" by method - Prove a specific statement
        # - thus/hence "statement" by method - Use previous facts
        # - proof - - Enter nested prove mode
        #
        # NEVER use 'apply' commands in state mode."""
        #         else:
        #             mode_instructions = """Valid commands in PROVE MODE:
        # - apply simp, apply auto, apply blast - Basic automation
        # - apply (simp add: lemma) - Simplify with specific lemmas
        # - using assms apply method - Use assumptions
        # - apply (induct x) - Induction on variable x
        # - proof - - Switch to structured proof mode
        #
        # NEVER use 'have' or 'show' commands in prove mode."""

        mode_instructions = self._get_mode_instructions(mode)

        # TODO: remove those?
        # examples_str = self._get_relevant_examples(mode)
        # failure_context = self._build_failure_context(parent_state)

        response = suggest_next_tactic(
            theorem=self.theorem,
            proof_state=parent_state.result,
            mode=mode,
            mode_description=mode_description,
            proof_history=proof_history_description,
            failed_attempts=failed_attempts_description,
            available_lemmas=available_lemmas,
            mode_specific_instructions=mode_instructions,
        )

        content = response.content
        print(f"\nReasoning:\n{content}\n")

        # Look for TACTIC: line
        tactic = None
        for line in content.split("\n"):
            if line.strip().startswith("TACTIC:"):
                tactic = line.split("TACTIC:", 1)[1].strip()
                break
        if not tactic:
            print("No tactic found in response")
            return None

        tactic = tactic.strip()
        if tactic in tried_commands:
            print(f"Already tried: {tactic}")
            return None

        if not is_valid_isabelle_command(tactic):
            print(f"Invalid command: {tactic}")
            return None

        print(f"Trying suggested tactic: {tactic}")
        tried_commands.add(tactic)

        # Try the tactic
        candidates = []
        child = self._process_command(parent_state, tactic, 0.0, candidates)

        if child:
            print(f"✓ Tactic succeeded with p={child.probability:.3f}")
            return child
        else:
            print("✗ Tactic failed")
            return None

        # if not tactic:
        #     print("No tactic found in response")
        #     return None
        #
        #
        # if not suggested_command:
        #     print("No valid command extracted from self-ask reasoning")
        #     return None
        #
        # print(f"Self-ask suggested: {suggested_command}")
        # self._update_viz(f"Self-ask suggests: {suggested_command[:50]}")
        #
        # if suggested_command in tried_commands:
        #     print(f"    SKIPPING: Already tried {suggested_command}")
        #     return None
        #
        # if not is_valid_isabelle_command(suggested_command):
        #     print(f"    SKIPPING: Invalid command {suggested_command}")
        #     return None
        #
        # print(f"\nTrying self-ask suggestinos: {suggested_command}")
        #
        # child = self._process_command(
        #     parent_state=parent_state,
        #     command=suggested_command,
        #     temperature=0.0,
        #     all_candidates=[] # TODO: check this param
        # )
        # return child

    # def _build_self_ask_query(self, parent_state: ProofState) -> str:
    #     """Build the query string for self-ask reasoning"""
    #     query_parts = [f"Theorem: }"]

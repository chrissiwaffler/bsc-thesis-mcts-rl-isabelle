import asyncio
import contextlib
import json
import os
import time
from datetime import datetime

import lilypad
from dotenv import load_dotenv

from .core import BeamSearchConfig, ProofGraph, QIsabelleSession
from .llm import IsabelleProofAgent
from .utils import (
    ProofTreeVisualizer,
    create_results_dir,
    extract_imports,
    extract_theorem_statement,
    finalize_proof,
    load_minif2f_files,
    save_result,
)

load_dotenv()


# lilypad for tracking primarly LLM calls and also some other stuff (currently: executing tacts)
lilypad.configure(
    auto_llm=True,
    api_key=os.environ.get("LILYPAD_API_KEY"),
    project_id=os.environ.get("LILYPAD_PROJECT_ID"),
)

# Check if API key is available
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment variables.")

os.environ["MIRASCOPE_DOCSTRING_PROMPT_TEMPLATE"] = "ENABLED"


@lilypad.trace(name="Solve Problem")
async def run_beam_search_minif2f(
    base_path="miniF2F/isabelle",
    split="valid",
    max_problems=10,
    max_depth=20,
    beam_width=5,
    temperatures=None,
    timeout_per_problem=600,  # 10 minutes
    port=17000,
    visualize=False,  # whether to show visualizations
    reuse_session=False,  # reuse isabelle session across problems
):
    """Run beam search algorithm on miniF2F problems

    Note: When running with Docker, you may encounter 'mkfifo' errors when creating
    multiple sessions. Set reuse_session=True to use a single session, but be aware
    that theory management between problems can be tricky."""

    if temperatures is None:
        temperatures = [0.3, 0.6, 0.9, 1.2]

    # init run
    # create results directory
    results_dir = create_results_dir("beam_search")
    print(f"Results will be saved to: {results_dir}")

    problems = load_minif2f_files(base_path, split, limit=max_problems)

    summary = {
        "total_problems": len(problems),
        "successful": 0,
        "failed": 0,
        "results": [],
        "start_time": datetime.now().isoformat(),
        "beam_config": {
            "beam_width": beam_width,
            "temperatures": temperatures,
            "max_depth": max_depth,
        },
    }

    # create single session if reusing
    shared_session = None
    if reuse_session:
        try:
            shared_session = QIsabelleSession(
                session_name="HOL",
                session_roots=[],
                port=port,
                debug=True,
            )
            print("Created shared Isabelle session for all problems")
        except Exception as e:
            print(f"Failed to create shared session: {e}")
            print("Will try creating individual sessions per problem")
            reuse_session = False

    # run each problem
    for i, problem in enumerate(problems):
        print(f"\n{'=' * 60}")
        print(f"Problem {i + 1}/{len(problems)}: {problem['name']}")
        print(f"{'=' * 60}")

        # Add metadata for this problem using a span
        with lilypad.span("Problem Metadata") as span:
            span.metadata(
                {
                    "problem_name": problem["name"],
                    "problem_index": i + 1,
                    "total_problems": len(problems),
                    "split": split,
                    "max_depth": max_depth,
                    "beam_width": beam_width,
                    "temperatures": temperatures,
                }
            )

        start_time = time.time()
        session = None
        viz = None

        try:
            # Use shared session or create new one
            if reuse_session and shared_session:
                session = shared_session
                print(f"Using shared Isabelle session for problem {problem['name']}")
            else:
                try:
                    session = QIsabelleSession(
                        session_name="HOL",
                        session_roots=[],
                        port=port,
                        debug=True,
                    )
                    print(f"Created new Isabelle session for problem {problem['name']}")
                except Exception as e:
                    print(f"Failed to create session: {e}")
                    print("Skipping this problem")
                    summary["failed"] += 1
                    error_result = save_result(
                        results_dir,
                        problem["name"],
                        False,
                        f"(* Session creation error: {e!s} *)",
                        {"error": f"Session creation failed: {e!s}"},
                        0.0,
                    )
                    summary["results"].append(error_result)
                    continue

            # regex extraction
            theorem = extract_theorem_statement(problem["content"])
            imports = extract_imports(problem["content"])

            # cleanup previous theory if reusing session
            if reuse_session and shared_session and i > 0:
                try:
                    # forget all states from previous problem
                    session.forget_all_states()
                    print("Cleaned up previous theory states")
                except Exception as e:
                    print(f"Warning: Failed to cleanup states: {e}")

            # init new theory
            try:
                session.new_theory(
                    theory_name=f"Test_{problem['name']}",
                    new_state_name=session.initial_state_name,
                    imports=imports,
                    only_import_from_session_heap=False,  # Allow dynamic loading
                )
            except Exception as e:
                print(f"Failed with imports {imports}: {e}")
                print("Falling back to Main import...")
                session.new_theory(
                    theory_name=f"Test_{problem['name']}",
                    new_state_name=session.initial_state_name,
                    imports=["Main"],
                    only_import_from_session_heap=False,
                )

            # init data structures
            graph = ProofGraph()
            viz = ProofTreeVisualizer(show_probabilities=True) if visualize else None

            # create root
            root = graph.add_state(theorem)
            is_done, result = session.execute(
                session.initial_state_name, theorem, root.state_name
            )
            graph.update_result(root.state_name, is_done, result)
            root.probability = 1.0  # root starts with probability 1

            if viz:
                viz.update(graph, f"Problem: {problem['name']}")

            # create agent
            agent = IsabelleProofAgent(
                graph=graph,
                session=session,
                viz=viz,
                theorem=theorem,
                config=BeamSearchConfig(
                    beam_width=beam_width,
                    temperatures=temperatures,
                    seed=42,
                ),
            )

            # run beam search with timeout
            final_state = None
            try:
                # timeout mechanism
                async def run_with_timeout(agent=agent, root=root, max_depth=max_depth):
                    return await agent.search(root, max_depth=max_depth)

                final_state = await asyncio.wait_for(
                    run_with_timeout(), timeout=timeout_per_problem
                )

            except asyncio.TimeoutError:
                print(f">Timeout for {problem['name']}")
                final_state = None

            elapsed = time.time() - start_time

            if final_state:
                # proof reconstruction & checking for validity
                success, just_proof, message = finalize_proof(
                    session, graph, theorem, final_state.state_name
                )
                if success:
                    # replace 'sorry' with actual proof in the original content
                    proof_text = problem["content"].replace("sorry", just_proof)
                else:
                    # use original content with sorry
                    proof_text = problem["content"]
            else:
                success = False
                proof_text = problem["content"]  # keep original with sorry
                message = "No proof found"

            # gather stats
            beam_sizes = []
            for depth_nodes in graph.beam_history:
                beam_sizes.append(len(depth_nodes))

            stats = {
                "nodes_explored": graph.G.number_of_nodes(),
                "max_depth": (
                    max(graph.get_state(n).depth for n in graph.G.nodes())
                    if graph.G.nodes()
                    else 0
                ),
                "beam_sizes": beam_sizes,
                "final_state": final_state.state_name if final_state else None,
                "message": message,
            }

            # save result
            result_data = save_result(
                results_dir,
                problem["name"],
                success,
                proof_text,
                stats,
                elapsed,
            )

            summary["results"].append(result_data)
            if success:
                summary["successful"] += 1
                print(f"SUCCESS in {elapsed:.2f}s")
            else:
                summary["failed"] += 1
                print(f"FAILED in {elapsed:.2f}s")

        except Exception as e:
            print(f"X ERROR: {e!s}")
            summary["failed"] += 1

            # save error res
            error_result = save_result(
                results_dir,
                problem["name"],
                False,
                f"(* Error: {e!s} *)",
                {"error": str(e)},
                time.time() - start_time,
            )
            summary["results"].append(error_result)

        finally:
            # cleanup
            if viz:
                viz.close()
            # only close session if not reusing
            if session and not (reuse_session and session == shared_session):
                with contextlib.suppress(Exception):
                    session.__exit__(None, None, None)

    # cleanup shared session if used
    if shared_session:
        try:
            shared_session.__exit__(None, None, None)
            print("Closed shared Isabelle session")
        except Exception as e:
            print(f"Error closing shared session: {e}")

    # save summary
    summary["end_time"] = datetime.now().isoformat()
    summary["total_time"] = sum(r["elapsed_time"] for r in summary["results"])

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Log final summary
    success_rate = (
        100 * summary["successful"] / summary["total_problems"]
        if summary["total_problems"] > 0
        else 0
    )

    print(
        f"\n{'=' * 60}\nBENCHMARK COMPLETE\n{'=' * 60}\n"
        f"Total problems: {summary['total_problems']}\n"
        f"Successful: {summary['successful']} ({success_rate:.1f}%)\n"
        f"Failed: {summary['failed']}\n"
        f"Total time: {summary['total_time']:.1f}s\n"
        f"Results saved to: {results_dir}"
    )

    return summary


async def run_beam_search_example():
    """example runner for beam search"""

    isabelle_file = """
    (*
  Authors: Wenda Li
*)

theory mathd_algebra_536
  imports Complex_Main "HOL-Computational_Algebra.Computational_Algebra"
    "HOL-Number_Theory.Number_Theory"
begin

theorem mathd_algebra_536:
  "fact 3 * (2^3 + sqrt 9) / 2 = 33"
  sorry

end
    """

    theorem = extract_theorem_statement(isabelle_file)
    imports = extract_imports(isabelle_file)

    print(f"theorem: {theorem}")
    print(f"imports: {imports}")

    # init session with HOL
    session = QIsabelleSession(
        session_name="HOL",
        session_roots=[],
        port=17000,
        debug=True,
    )
    print("connected to isabelle server")

    # For this theorem, we'll use the approach shown in the README:
    # Import theories that aren't in the heap by setting only_import_from_session_heap=False
    # This allows dynamic loading of theories

    # Try to create theory with the extracted imports
    print(f"\nCreating theory with imports: {imports}")
    try:
        session.new_theory(
            theory_name="BeamSearchTest",
            new_state_name=session.initial_state_name,
            imports=imports,
            only_import_from_session_heap=False,  # Allow dynamic loading
        )
        print(f"Successfully created theory with imports: {imports}")
    except Exception as e:
        print(f"Failed to create theory with full imports: {e}")
        print("Trying with just Complex_Main...")
        try:
            session.new_theory(
                theory_name="BeamSearchTest",
                new_state_name=session.initial_state_name,
                imports=["Complex_Main"],
                only_import_from_session_heap=False,
            )
            print("Successfully created theory with Complex_Main")
        except Exception as e2:
            print(f"Failed with Complex_Main: {e2}")
            print("Falling back to Main import...")
            session.new_theory(
                theory_name="BeamSearchTest",
                new_state_name=session.initial_state_name,
                imports=["Main"],
                only_import_from_session_heap=False,
            )

    # init data structures
    graph = ProofGraph()
    viz = ProofTreeVisualizer(show_probabilities=True)

    # create root using the extracted theorem
    root = graph.add_state(theorem)
    is_done, result = session.execute(
        session.initial_state_name, theorem, root.state_name
    )
    graph.update_result(root.state_name, is_done, result)
    root.probability = 1.0  # root starts with probability 1

    viz.update(graph, "initial state")

    # create agent
    agent = IsabelleProofAgent(
        graph=graph,
        session=session,
        viz=viz,
        theorem=theorem,
        config=BeamSearchConfig(
            beam_width=5,
            temperatures=[0.3, 0.6, 0.9, 1.2],
            seed=42,
        ),
    )

    # run beam search
    final_state = await agent.search(root, max_depth=20)

    if final_state:
        # reconstruct and verify
        success, proof_text, message = finalize_proof(
            session,
            graph,
            theorem,
            final_state.state_name,
        )

        if success:
            print("\nFINAL VERIFIED PROOF:")
            print(f"{theorem}")
            print(proof_text)
        else:
            print(f"\nRECONSTRUCTION FAILED: {message}")
    else:
        print("NO PROOF FOUND!")

    # show final visualization
    viz.show()


if __name__ == "__main__":
    # run beam search on miniF2F benchmark
    asyncio.run(
        run_beam_search_minif2f(
            base_path="miniF2F/isabelle",
            split="test",  # or "valid"
            max_problems=50,
            max_depth=20,
            beam_width=5,
            temperatures=[0.3, 0.7, 0.9, 1.2, 1.5],
            timeout_per_problem=1200,  # 20 minutes per problem
            port=17000,
            visualize=False,
        )
    )

    # asyncio.run(run_beam_search_example())

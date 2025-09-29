few_shot_examples = [
    # Examples for prove mode - using apply tactics
    {
        "question": "Theorem: fixes x :: complex shows x^2 + 49 = (x + 7*I) * (x - 7*I)\nCurrent goals: 1. x^2 + 49 = (x + 7*I) * (x - 7*I)\nCurrent mode: prove",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What is the structure of the goal equation?
Intermediate answer: The goal is to prove an equality between x^2 + 49 and a product of complex conjugates.
Follow up: Should I switch to structured proof mode with 'proof -' to work through this step by step?
Intermediate answer: No, this is a straightforward algebraic identity that can be solved directly.
Follow up: Can I expand the right-hand side to match the left?
Intermediate answer: Yes, (x + 7*I) * (x - 7*I) = x^2 - (7*I)^2 = x^2 - 49*I^2 = x^2 + 49 since I^2 = -1.
Follow up: What tactic handles algebraic simplification and complex arithmetic?
Intermediate answer: The 'simp' tactic with algebra_simps can handle this expansion.
So the final answer is: apply (simp add: algebra_simps)""",
    },
    # Example with multiple named assumptions for linear arithmetic
    {
        "question": "Theorem: fixes a b :: real assumes h0: 3*a + 2*b = 12 and h1: a = 4 shows b = 0\nCurrent goals: 1. b = 0\nCurrent mode: prove",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What are the given assumptions?
Intermediate answer: h0 states that 3*a + 2*b = 12, and h1 states that a = 4.
Follow up: Can I substitute h1 into h0 to solve for b?
Intermediate answer: Yes, substituting a = 4 into 3*a + 2*b = 12 gives 12 + 2*b = 12, so b = 0.
Follow up: What tactic can use both assumptions together for linear arithmetic?
Intermediate answer: The linarith tactic is designed for linear arithmetic and can use multiple assumptions.
Follow up: How do I give linarith access to both h0 and h1?
Intermediate answer: Use 'using h0 h1 apply linarith' to provide both assumptions to linarith.
So the final answer is: using h0 h1 apply linarith""",
    },
    {
        "question": "Theorem: fixes a b :: nat assumes a_le_b: a ≤ b shows a + 1 ≤ b + 1\nCurrent goals: 1. a + 1 ≤ b + 1\nCurrent mode: prove\nAvailable lemmas: a_le_b",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What kind of inequality is this?
Intermediate answer: This is about preserving inequality when adding the same value to both sides.
Follow up: Should I use 'proof -' to handle this step by step or try direct tactics?
Intermediate answer: This is simple enough for direct tactics - no need for structured proof mode.
Follow up: Can simp handle this directly?
Intermediate answer: Yes, simp knows basic arithmetic properties of inequalities.
Follow up: Do I need to reference the assumption explicitly?
Intermediate answer: It's safer to use the assumption explicitly with 'using'.
So the final answer is: apply (simp add: a_le_b)""",
    },
    {
        "question": 'Theorem: fixes f :: real ⇒ real assumes cont: continuous_on UNIV f shows ∃x. f x = x\nCurrent goals: 1. ∃x. f x = x\nCurrent mode: prove\nAvailable lemmas: cont\nFailed tactics: [("apply simp", "Failed to apply proof method"), ("apply auto", "Failed to apply proof method")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What type of theorem is this?
Intermediate answer: This looks like a fixed point theorem requiring continuity.
Follow up: Why did simp and auto fail?
Intermediate answer: They can't construct the witness for the existential statement.
Follow up: What specialized tactic might work?
Intermediate answer: This needs a specific theorem about continuous functions, not basic simplification.
Follow up: Should I switch to structured proof mode?
Intermediate answer: Yes, existential proofs often need explicit construction or specialized theorems.
So the final answer is: proof -""",
    },
    {
        "question": "Theorem: fixes x y :: real assumes x > 0 and y > 0 shows (x + y) / 2 ≥ sqrt (x * y)\nCurrent goals: 1. (x + y) / 2 ≥ sqrt (x * y)\nCurrent mode: prove\nAvailable lemmas: assms",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What inequality is this?
Intermediate answer: This is the AM-GM inequality (arithmetic mean ≥ geometric mean).
Follow up: Is this a standard lemma in Isabelle?
Intermediate answer: Yes, Isabelle likely has lemmas about AM-GM inequality.
Follow up: Should I try direct tactics or switch to 'proof -' for a structured approach?
Intermediate answer: Try direct tactics first since this is a well-known inequality.
Follow up: What tactic could apply such lemmas?
Intermediate answer: The arith tactic handles arithmetic inequalities.
So the final answer is: apply arith""",
    },
    # Examples showing multiple apply steps
    {
        "question": 'Theorem: fixes n :: nat shows n * (n + 1) = n^2 + n\nCurrent goals: 1. n * (n + 1) = n^2 + n\nCurrent mode: prove\nFailed tactics: [("apply simp", "Did not solve completely - remaining goal: n * Suc n = n * n + n")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What happened with apply simp?
Intermediate answer: It partially simplified but didn't complete the proof, leaving n * Suc n = n * n + n.
Follow up: What's the relationship between Suc n and n + 1?
Intermediate answer: Suc n is the successor function, equal to n + 1.
Follow up: Can I continue with another tactic?
Intermediate answer: Yes, since I used apply (not by), I can add more tactics.
Follow up: What tactic handles arithmetic with successor?
Intermediate answer: The algebra or ring_simps tactics understand successor arithmetic.
So the final answer is: apply (simp add: algebra_simps)""",
    },
    {
        "question": 'Theorem: fixes xs :: nat list shows rev (rev xs) = xs\nCurrent goals: 1. rev (rev xs) = xs\nCurrent mode: prove\nFailed tactics: [("apply simp", "Failed to apply proof method")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: Why might simp have failed on this standard property?
Intermediate answer: This property might need induction on the list structure.
Follow up: What's the standard approach for list properties?
Intermediate answer: Induction on the list: either xs = [] or xs = y # ys.
Follow up: Should I switch to structured proof mode with 'proof -' or try induction with apply?
Intermediate answer: For simple induction, apply (induct xs) should work well enough.
Follow up: When would 'proof -' be better than apply (induct xs)?
Intermediate answer: When I need to handle complex induction cases with intermediate lemmas.
So the final answer is: apply (induct xs)""",
    },
    # NEW: Example showing when proof - is clearly better
    {
        "question": 'Theorem: fixes f g :: "nat ⇒ real" assumes "∀n. f n > 0" and "∀n. g n = f n + f (n+1)" shows "∀n. g n > f n"\nCurrent goals: 1. ∀n. g n > f n\nCurrent mode: prove\nFailed tactics: [("apply simp", "Failed"), ("apply (simp add: assms)", "Failed"), ("apply auto", "Failed")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What kind of proof is this?
Intermediate answer: This requires showing g n = f n + f(n+1) > f n for all n.
Follow up: Why did the simple tactics fail?
Intermediate answer: They can't automatically reason about the relationship between f n and f(n+1).
Follow up: Can I continue trying more apply tactics or should I switch to 'proof -'?
Intermediate answer: After 3 failures, and given the need to expand definitions, 'proof -' is better.
Follow up: What advantage does 'proof -' offer here?
Intermediate answer: I can fix n, expand g n using the assumption, and show the inequality step by step.
So the final answer is: proof -""",
    },
    # Example with completing the proof
    {
        "question": "Theorem: fixes a b c :: real assumes a = b and b = c shows a = c\nCurrent goals: 1. a = c\nCurrent mode: prove\nAvailable lemmas: assms",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What kind of proof is this?
Intermediate answer: This is transitivity of equality.
Follow up: Can simp handle transitivity with the given assumptions?
Intermediate answer: Yes, simp can chain equalities from assumptions.
Follow up: What tactic should I use?
Intermediate answer: Apply simp with the assumptions to complete the proof.
So the final answer is: apply (simp add: assms)""",
    },
    # Examples for switching between prove and state mode
    {
        "question": "Theorem: fixes x e :: complex assumes h0: x + e = 7 and h1: 2 * x + e = 3 shows e = 11 ∧ x = -4\nCurrent goals: 1. e = 11 ∧ x = -4\nCurrent mode: prove\nAvailable lemmas: h0, h1",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What kind of problem is this?
Intermediate answer: This is a system of two linear equations with two unknowns.
Follow up: Can I solve this directly with apply tactics like simp or auto?
Intermediate answer: No, this needs intermediate steps to solve for x and e separately.
Follow up: Should I switch to structured proof mode using 'proof -'?
Intermediate answer: Yes, structured proof mode allows me to show intermediate calculations step by step.
Follow up: Why is 'proof -' better than continuing with apply tactics here?
Intermediate answer: Because I need to derive intermediate facts (like x = -4) before proving the conjunction.
So the final answer is: proof -""",
    },
    # Examples for state mode - COMPLETE statements (stay in state mode)
    {
        "question": "Current mode: state\nproof (state)\nfix x e :: complex\nassume h0: x + e = 7\nassume h1: 2 * x + e = 3\nhave (2 * x + e) - (x + e) = 3 - 7\ngoal: 1. (2 * x + e) - (x + e) = 3 - 7",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in structured proof state mode.
Follow up: What do I need to prove?
Intermediate answer: That (2 * x + e) - (x + e) = 3 - 7.
Follow up: Can I use the assumptions h0 and h1?
Intermediate answer: Yes, h1 says 2 * x + e = 3 and h0 says x + e = 7.
Follow up: Is this simple enough to prove immediately?
Intermediate answer: Yes, it's just substitution and basic arithmetic.
So the final answer is: have "(2 * x + e) - (x + e) = 3 - 7\"""",
    },
    {
        "question": "Current mode: state\nproof (state)\nfix f :: real ⇒ real\nassume f_times: ∀x>0. ∀y>0. f(x*y) = f x / y\nassume f 500 = 3\nhave f 600 = f (500*(6/5))\nthis: f 600 = f (500*(6/5))\ngoal: 1. f 600 = f (500 * (6/5))",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode within a structured proof.
Follow up: What does the goal ask?
Intermediate answer: To show that f 600 = f (500*(6/5)).
Follow up: Is 600 = 500*(6/5)?
Intermediate answer: Yes, 500 * 6/5 = 500 * 1.2 = 600.
Follow up: Is this a simple arithmetic fact?
Intermediate answer: Yes, it's basic multiplication that can be verified automatically.
So the final answer is: have "f 600 = f (500*(6/5))\"""",
    },
    # Examples for state mode - INCOMPLETE statements (switch to prove mode)
    {
        "question": 'Current mode: state\nproof (state)\nfix a b c :: real\nassume abc: "a^2 + b^2 = c^2"\nassume a_pos: "a > 0" and b_pos: "b > 0"\nhave "c > 0"\ngoal: 1. c > 0',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode and need to prove c > 0.
Follow up: Is this proof immediate from the assumptions?
Intermediate answer: No, I need to show that c^2 = a^2 + b^2 > 0, then take square root.
Follow up: Should I complete this proof immediately or work through it step by step?
Intermediate answer: This requires careful reasoning about squares and positivity, better to work step by step.
So the final answer is: proof -""",
    },
    {
        "question": 'Current mode: state\nproof (state)\nfix n :: nat\nassume n_gt: "n > 2"\nhave "∃p q. p < n ∧ q < n ∧ n = p + q"\ngoal: 1. ∃p q. p < n ∧ q < n ∧ n = p + q',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode, trying to prove existence of a decomposition.
Follow up: Can I prove this with a simple tactic?
Intermediate answer: No, I need to construct specific values for p and q.
Follow up: What's a good approach?
Intermediate answer: I could choose p = 1 and q = n - 1, then verify the conditions.
Follow up: Should I defer to prove mode to construct the witnesses?
Intermediate answer: Yes, existential proofs often need explicit construction.
So the final answer is: proof -""",
    },
    # Mixed example - showing labeled statements
    {
        "question": 'Current mode: state\nproof (state)\nfix x y z :: real\nassume xyz: "x * y * z = 1"\nhave xy_relation: "z = 1 / (x * y)"\ngoal: Need to establish relationships',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What did I just do?
Intermediate answer: I made a labeled statement xy_relation without immediate proof.
Follow up: Why wasn't I forced to prove it immediately?
Intermediate answer: Because I used a label, Isabelle defers the proof until later.
Follow up: Should I prove it now or continue with more setup?
Intermediate answer: Since the goal says "establish relationships", I should prove this one first.
Follow up: Is this provable from xyz?
Intermediate answer: Yes, from x*y*z = 1, we get z = 1/(x*y) by dividing both sides.
So the final answer is: have xy_relation: "z = 1 / (x * y)\"""",
    },
    # Example showing when NOT to defer
    {
        "question": 'Current mode: state\nproof (state)\nfix a b :: nat\nassume "a ≤ b"\nhave "a ≤ b + 1"\ngoal: 1. a ≤ b + 1',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode, proving a ≤ b + 1.
Follow up: Is this complex to prove?
Intermediate answer: No, if a ≤ b, then clearly a ≤ b + 1.
Follow up: Should I defer this to prove mode?
Intermediate answer: No, this is trivial and deferring would be overkill.
Follow up: What's the simplest way to prove it?
Intermediate answer: Use the assumption and basic arithmetic.
So the final answer is: have "a ≤ b + 1\"""",
    },
    # Examples showing mode transitions
    {
        "question": "Theorem: fixes a b :: real assumes b ≤ a and 1 < b shows ln(a/b)/ln a + ln(b/a)/ln b ≤ 0\nCurrent goals: 1. ln(a/b)/ln a + ln(b/a)/ln b ≤ 0\nCurrent mode: prove",
        "answer": """Are follow up questions needed here: Yes.
Follow up: Is this expression complex to work with directly?
Intermediate answer: Yes, the logarithms make it complicated.
Follow up: Can I try apply tactics first or should I immediately switch to 'proof -'?
Intermediate answer: The complexity suggests 'proof -' would be better from the start.
Follow up: Would variable substitution help?
Intermediate answer: Yes, let x = ln a and y = ln b to simplify.
Follow up: Can I do variable substitution with apply tactics?
Intermediate answer: No, defining new variables requires structured proof mode.
Follow up: So should I switch to structured proof mode with 'proof -'?
Intermediate answer: Yes, I need to define new variables and prove properties about them.
So the final answer is: proof -""",
    },
    {
        "question": "Current mode: state\nproof (state)\ndefine x y where x = ln a and y = ln b\nthis: x = ln a ∧ y = ln b\nassume b ≤ a\nassume 1 < b\ngoal: 1. ?thesis",
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode after defining variables.
Follow up: What properties can I derive about x and y?
Intermediate answer: Since b > 1, we have y = ln b > 0. Since b ≤ a, we have y ≤ x.
Follow up: Should I prove these properties with 'have' statements?
Intermediate answer: Yes, I need to establish y > 0 and x ≥ y.
Follow up: Is y > 0 simple to prove?
Intermediate answer: It needs the fact that ln is positive for arguments > 1, so I'll complete it immediately.
So the final answer is: have "y > 0\"""",
    },
    # Example showing multiple connected have statements
    {
        "question": 'Current mode: state\nproof (state)\nfix x e :: complex\nassume h0: x + e = 7\nassume h1: 2 * x + e = 3\nhave eq1: "(2 * x + e) - (x + e) = 3 - 7"\nthis: (2 * x + e) - (x + e) = 3 - 7\ngoal: Continue derivation',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode with one equation already established.
Follow up: What's the next logical step in solving for x and e?
Intermediate answer: I should simplify the left side to get x = -4.
Follow up: Since I already have one equation, how should I continue?
Intermediate answer: Use "also have" to connect this to the previous statement.
So the final answer is: also have "x = -4\"""",
    },
    {
        "question": 'Current mode: state\nproof (state)\nfix a b c :: real\nassume a_pos: "a > 0" and b_pos: "b > 0" and c_pos: "c > 0"\nassume sum_eq: "a + b + c = 1"\nhave step1: "a < 1"\nthis: a < 1\ngoal: Continue establishing bounds',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode with one inequality established.
Follow up: What other bounds can I derive?
Intermediate answer: Since a + b + c = 1 and all are positive, b < 1 and c < 1 also hold.
Follow up: How should I state the next bound?
Intermediate answer: Use "also have" to connect to the previous statement about a.
So the final answer is: also have "b < 1\"""",
    },
    # Examples showing when to defer after failed attempts
    {
        "question": 'Current mode: state\nproof (state)\nfix x :: real\nassume x_pos: "x > 0"\nhave "ln(x^2) = 2 * ln x"\ngoal: 1. ln(x^2) = 2 * ln x\nFailed tactics: [("have \\"ln(x^2) = 2 * ln x\\" by simp", "Failed to finish proof"), ("have \\"ln(x^2) = 2 * ln x\\" by auto", "Failed to finish proof"), ("have \\"ln(x^2) = 2 * ln x\\" using x_pos by (simp add: ln_pow)", "Failed to finish proof")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode, trying to prove ln(x^2) = 2 * ln x.
Follow up: Have there been previous attempts to prove this?
Intermediate answer: Yes, three attempts failed - simp, auto, and trying to use ln_pow which doesn't exist.
Follow up: What does this pattern of failures suggest?
Intermediate answer: The automatic methods can't handle this logarithm property directly, and we need a more careful approach.
Follow up: Should I try another "by method" or defer the proof?
Intermediate answer: Given multiple failures, I should defer and work through it step by step in prove mode.
So the final answer is: have "ln(x^2) = 2 * ln x""",
    },
    {
        "question": 'Current mode: state\nproof (state)\nfix a b :: nat\nassume "a > 0" and "b > 0"\nhave gcd_comm: "gcd a b = gcd b a"\ngoal: 1. gcd a b = gcd b a\nFailed tactics: [("have gcd_comm: \\"gcd a b = gcd b a\\" by simp", "Failed to finish proof"), ("have gcd_comm: \\"gcd a b = gcd b a\\" by (simp add: gcd_commute)", "Failed to finish proof")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode with a labeled statement about gcd commutativity.
Follow up: What have the failed attempts shown?
Intermediate answer: Simple tactics don't work, and gcd_commute might not be the right lemma name.
Follow up: Since this is a labeled statement, what's the best approach?
Intermediate answer: With labeled statements, we can defer the proof and potentially use it later even without proving it immediately.
Follow up: Should I keep trying methods or defer?
Intermediate answer: After two failures, especially with a labeled statement, defer the proof.
So the final answer is: have gcd_comm: "gcd a b = gcd b a""",
    },
    {
        "question": 'Current mode: state\nproof (state)\nfix f :: "nat ⇒ real"\nassume mono_f: "∀n. f n ≤ f (Suc n)"\nhave "∀n m. n ≤ m ⟶ f n ≤ f m"\ngoal: 1. ∀n m. n ≤ m ⟶ f n ≤ f m\nFailed tactics: [("have \\"∀n m. n ≤ m ⟶ f n ≤ f m\\" using mono_f by auto", "Failed to finish proof"), ("have \\"∀n m. n ≤ m ⟶ f n ≤ f m\\" using mono_f by blast", "Failed to finish proof"), ("have \\"∀n m. n ≤ m ⟶ f n ≤ f m\\" using mono_f by (simp add: lift_Suc_mono_le)", "Failed to finish proof")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode, trying to prove a generalization of monotonicity.
Follow up: What do the failures tell us?
Intermediate answer: Auto and blast can't automatically generalize from single-step to arbitrary steps, and the lemma name was wrong.
Follow up: Is this the kind of proof that needs induction?
Intermediate answer: Yes, generalizing from f n ≤ f (Suc n) to arbitrary n ≤ m typically requires induction.
Follow up: Should I defer this to prove mode?
Intermediate answer: Yes, after three failures with different automatic methods, this clearly needs structured inductive reasoning.
So the final answer is: have "∀n m. n ≤ m ⟶ f n ≤ f m""",
    },
    # Example showing when to just state without proof after errors
    {
        "question": 'Current mode: state\nproof (state)\nfix x :: real\nassume "x ≠ 0"\nhave "x * (1/x) = 1"\ngoal: 1. x * (1/x) = 1\nFailed tactics: [("have \\"x * (1/x) = 1\\" by simp", "Failed to finish proof")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode, proving x * (1/x) = 1.
Follow up: Why might simp have failed?
Intermediate answer: Simp might not know to use the assumption x ≠ 0 automatically.
Follow up: Is this fundamentally a difficult proof?
Intermediate answer: No, it's basic algebra, just needs the right approach.
Follow up: Should I defer or state it differently?
Intermediate answer: After a failure, it's often better to just state the fact without trying to prove it immediately.
So the final answer is: have "x * (1/x) = 1\"""",
    },
    # Example showing multiple failures suggest just stating the fact
    {
        "question": 'Current mode: state\nproof (state)\nfix p q :: "nat ⇒ bool"\nassume pq_equiv: "∀n. p n ⟷ q n"\nhave "∃n. p n ⟷ ∃n. q n"\ngoal: 1. (∃n. p n) ⟷ (∃n. q n)\nFailed tactics: [("have \\"(∃n. p n) ⟷ (∃n. q n)\\" using pq_equiv by auto", "Failed to finish proof"), ("have \\"(∃n. p n) ⟷ (∃n. q n)\\" using pq_equiv by blast", "Failed to finish proof"), ("have \\"(∃n. p n) ⟷ (∃n. q n)\\" using pq_equiv by force", "Failed to finish proof")]',
        "answer": """Are follow up questions needed here: Yes.
Follow up: What mode am I in?
Intermediate answer: I'm in state mode trying to prove an equivalence of existential statements.
Follow up: What pattern do I see in the failures?
Intermediate answer: Multiple automatic methods (auto, blast, force) all failed, even with the relevant assumption.
Follow up: What does this suggest about the proof?
Intermediate answer: The proof likely needs more careful reasoning than automatic methods can provide.
Follow up: What's the best approach after multiple failures?
Intermediate answer: In state mode, when multiple proof attempts fail, it's better to just state the fact and defer the proof.
So the final answer is: have "(∃n. p n) ⟷ (∃n. q n)\"""",
    },
]

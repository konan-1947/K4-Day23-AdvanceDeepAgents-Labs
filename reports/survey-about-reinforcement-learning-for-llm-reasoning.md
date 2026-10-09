# Reinforcement Learning for Large Language Model Reasoning: Paradigms, Algorithms, and Test-Time Compute

## TL;DR
- Reinforcement learning (RL) has transitioned from superficial style alignment [1][2] to a primary mechanism for eliciting verified multi-step reasoning, self-reflection, and mathematical problem-solving [3][4].
- Outcome-based reward models (ORM) struggle with false positives, making process-supervised reward models (PRM) [5][6] and rule-based verifiable reward signals [3][4] crucial for credit assignment across extended rationales.
- Group Relative Policy Optimization (GRPO) eliminates the memory-heavy critic network of PPO by computing advantage estimates across sampled response cohorts, dramatically lowering training overhead [3][4].
- Pure RL with rule-based rewards produces emergent test-time deliberation behaviors, including backtracking, counter-example generation, and dynamic chain-of-thought expansion [4][7], complementing test-time compute scaling [8].

## Background
Reinforcement learning with human feedback (RLHF) was originally conceived to steer model behavior toward helpfulness and safety using Proximal Policy Optimization (PPO) over paired comparisons [1]. Later formulations like Direct Preference Optimization (DPO) bypassed separate reward models by analytically optimizing a closed-form implicit objective directly on static preference datasets [2]. However, preference optimization algorithms inherently falter on multi-step reasoning tasks such as formal mathematics, symbolic logic, and algorithmic coding [3]. In these objective domains, human subjective preference is irrelevant compared to mathematical validity. Consequently, research pivoted toward reinforcement learning with verifiable reward signals (RLVR), where ground-truth verification and automated test suites provide objective reward functions [3][4][7].

## Credit Assignment: Outcome Supervision vs. Process Supervision
A primary bottleneck in training LLMs on complex rationales is credit assignment across reasoning steps. Standard outcome reward models (ORM) evaluate only the final answer, rewarding deceptive rationales that arrive at correct conclusions through flawed steps [5]. To counter this, process supervision provides step-level feedback through Process Reward Models (PRMs) trained on fine-grained human or automated step verifications [5][6]. Active search with PRM guidance substantially boosts pass rates across MATH and GSM8K benchmarks compared to majority voting over outcome models [6]. Furthermore, reverse curriculum reinforcement learning demonstrates that initializing rollouts close to verified intermediate sub-goals prevents catastrophic policy collapse during early training stages [9].

## Algorithmic Innovation: From PPO to GRPO and Self-Evolving Reasoning
Standard PPO requires maintaining four concurrent model replicas in GPU memory: policy, reference model, value/critic network, and reward model [1][3]. This memory footprint severely constrains context length during long-horizon reasoning rollouts [3]. DeepSeekMath introduced Group Relative Policy Optimization (GRPO), which removes the value network entirely by sampling a group of outputs for each question and standardizing their rewards to form relative baseline advantages [3]. Building on this, DeepSeek-R1 demonstrated that large-scale RL using rule-based accuracy and formatting rewards (R1-Zero) induces self-reflection, automatic error correction, and long chains-of-thought without supervised fine-tuning data [4]. Models autonomously discover "aha moments," spontaneously allocating more inference tokens to verify ambiguous calculation steps [4][7].

## Test-Time Compute and Inference-Time Search
The convergence of RL training with inference-time search reveals that reasoning capability scales along the compute dimension at inference time as well as pretraining [8][7]. Rather than generating greedy completions, systems leverage trained policy verifiers and search trees (such as Monte Carlo Tree Search or beam search) to explore multiple reasoning paths [8]. Empirical findings demonstrate that optimally allocating test-time compute across verifiable candidates can surpass the performance of an order-of-magnitude larger base model trained solely on next-token prediction [8].

## Trends and open problems
The field of RL for reasoning is rapidly developing around several open questions. First, reward hacking remains prevalent when reward functions rely on imperfect learned neural verifiers rather than formal syntax or deterministic sandboxes [5][3]. Second, language drift and verbosity inflation frequently occur: RL agents naturally discover that generating verbose rationales artificially inflates output likelihood under certain heuristics even when the underlying reasoning is redundant [4]. Third, generalizing verifiable RL beyond formal STEM disciplines (math and coding) to open-ended synthesis, philosophical reasoning, and legal analysis remains challenging due to the absence of verifiable ground-truth oracles [3][4].

## References
[1] Training language models to follow instructions with human feedback. arxiv. https://arxiv.org/abs/2203.02155 (2022-03-04)
[2] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arxiv. https://arxiv.org/abs/2305.18290 (2023-05-29)
[3] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arxiv. https://arxiv.org/abs/2402.03300 (2024-02-05)
[4] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. hf-daily. https://huggingface.co/papers/2501.12948 (2025-01-22)
[5] Solving math word problems with process- and outcome-based feedback. arxiv. https://arxiv.org/abs/2211.14275 (2022-11-25)
[6] Let us Verify Step by Step. arxiv. https://arxiv.org/abs/2305.20050 (2023-05-31)
[7] Learning to Reason with LLMs. web. https://openai.com/index/learning-to-reason-with-llms (2024-09-12)
[8] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Parameters. arxiv. https://arxiv.org/abs/2408.03314 (2024-08-06)
[9] Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning. hf-search. https://huggingface.co/papers/2402.01980 (2024-02-02)
